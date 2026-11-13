import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer


# ---------- module-level helpers (no SteeredModel coupling) ----------

@torch.no_grad()
def get_neuron_values(sm, L, n, strength=1.0, calibrated=True):
    """Normalized down-column for neuron n at layer L, scaled by strength.
    calibrated=True multiplies by median_norm(L) so strength is in residual-norm units."""
    dp = sm.layers[L].mlp.down_proj
    base = dp.base if isinstance(dp, LoRATweak) else dp
    e = torch.zeros(1, base.in_features, dtype=torch.bfloat16, device=base.weight.device)
    e[0, n] = 1.0
    col = base(e)[0].float()
    norm = col.norm()
    assert norm > 0, f"zero down-column at L{L} n{n}; check device / dtype"
    v = col / norm
    if calibrated:
        v = v * sm.median_norm(L)
    return v * strength


def add_hook_steer(model_layers, L, vec):
    """One-off hook that adds vec to down_proj's output. Returns the handle."""
    v = vec.detach().clone()
    def hook(mod, inp, out):
        return out + v.to(device=out.device, dtype=out.dtype)
    return model_layers[L].mlp.down_proj.register_forward_hook(hook)


# ---------- the wrapper ----------

class LoRATweak(nn.Module):
    """y = Wx + (B@A)x * lora_scale + bias. A/B are trainable; bias is hand-set."""
    def __init__(self, base, bias, r=8, alpha=16, compute_dtype=torch.bfloat16):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.lora_scale = alpha / r
        d_out, d_in = base.out_features, base.in_features
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(r, d_in, dtype=compute_dtype, device=dev) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=compute_dtype, device=dev))
        self.bias = nn.Parameter(bias.to(device=dev, dtype=compute_dtype).clone())

    def forward(self, x):
        out = self.base(x)
        xa = x.to(self.A.dtype)
        lora = ((xa @ self.A.T) @ self.B.T) * self.lora_scale
        return out + lora.to(out.dtype) + self.bias.to(out.dtype)

    def trainable(self, mode="lora"):
        if mode == "lora": return [self.A, self.B]
        if mode == "bias": return [self.bias]
        if mode == "both": return [self.A, self.B, self.bias]
        raise ValueError(mode)


# ---------- dataset ----------

class ChatDataset(Dataset):
    """conversations: list of [[role, text], ...]. Trains only on 'assistant' turns."""
    def __init__(self, conversations, tok, max_len=256):
        self.convs, self.tok, self.max_len = conversations, tok, max_len

    def __len__(self): return len(self.convs)

    def __getitem__(self, i):
        turns = [{"role": r, "content": t} for r, t in self.convs[i]]
        full = self.tok.apply_chat_template(
            turns, add_generation_prompt=False, return_tensors="pt",
            return_dict=True, enable_thinking=False)
        ids = full["input_ids"][0][:self.max_len]
        labels = torch.full_like(ids, -100)
        for k, (role, text) in enumerate(self.convs[i]):
            if role != "assistant": continue
            prefix = self.tok.apply_chat_template(
                turns[:k], add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False)["input_ids"][0]
            seg = self.tok(text, add_special_tokens=False)["input_ids"]
            start = len(prefix)
            end = min(start + len(seg), self.max_len)
            labels[start:end] = ids[start:end]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


# ---------- the model ----------

class SteeredModel:
    def __init__(self, model_id="Qwen/Qwen3.5-4B", calib_prompts=None,
                 max_memory=None, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=q, device_map="auto",
            max_memory=max_memory or {0: "5GiB", "cpu": "16GiB"}, dtype=dtype)
        self.layers, self.layer_path = self._find_layers()
        self.tweaks = {}          # layer -> LoRATweak
        self._med = {}
        self._calib = calib_prompts or [
            "The quick brown fox jumps over the lazy dog.",
            "Explain photosynthesis in one sentence.",
            "What is the capital of France?",
            "Write a haiku about rain.",
        ]

    # ---- layout ----
    def _find_layers(self):
        for path in ["model.layers", "model.language_model.layers",
                     "language_model.model.layers", "layers"]:
            obj = self.model
            try:
                for p in path.split("."): obj = getattr(obj, p)
            except AttributeError:
                continue
            if isinstance(obj, nn.ModuleList) and hasattr(obj[0], "mlp"):
                return obj, path
        raise RuntimeError("could not find decoder layer list")

    # ---- calibration ----
    @torch.no_grad()
    def median_norm(self, L, exclude_first_token=True):
        if L in self._med: return self._med[L]
        norms = []
        for t in self._calib:
            enc = self.tok(t, return_tensors="pt").to(self.model.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            h = hs[L + 1][0]
            if exclude_first_token: h = h[1:]
            norms.append(h.float().norm(dim=-1))
        self._med[L] = torch.cat(norms).median().item()
        return self._med[L]

    # ---- wrapping ----
    def wrap(self, *layers, r=8, alpha=16, bias=None, dtype=torch.bfloat16):
        """Wrap each layer's down_proj with a LoRATweak. bias defaults to zeros.
        Returns dict {layer: LoRATweak}; also stored in self.tweaks. Chainable state."""
        out = {}
        for L in layers:
            dp = self.layers[L].mlp.down_proj
            if isinstance(dp, LoRATweak):
                raise RuntimeError(f"layer {L} already wrapped")
            d_out = dp.out_features
            b = torch.zeros(d_out, dtype=torch.float32, device=dp.weight.device) if bias is None else bias
            w = LoRATweak(dp, b, r=r, alpha=alpha, compute_dtype=dtype)
            self.layers[L].mlp.down_proj = w
            self.tweaks[L] = w
            out[L] = w
        return out

    def set_neuron_bias(self, L, n, strength=1.0, calibrated=True):
        """Write the neuron's down-column into sm.tweaks[L].bias. Wrap first."""
        if L not in self.tweaks:
            raise RuntimeError(f"wrap({L}) first")
        v = get_neuron_values(self, L, n, strength=strength, calibrated=calibrated)
        self.tweaks[L].bias.data.copy_(v.to(self.tweaks[L].bias.dtype))

    # ---- training ----
    def freeze_base(self):
        for p in self.model.parameters(): p.requires_grad_(False)
        for w in self.tweaks.values():
            w.A.requires_grad_(False); w.B.requires_grad_(False); w.bias.requires_grad_(False)

    def trainable(self, mode="lora"):
        ps = []
        for w in self.tweaks.values(): ps += w.trainable(mode)
        for p in ps: p.requires_grad_(True)
        return ps

    def _collate(self, batch):
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        maxlen = max(b["input_ids"].size(0) for b in batch)
        ids  = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        lbls = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        for j, b in enumerate(batch):
            n = b["input_ids"].size(0)
            ids[j, :n]  = b["input_ids"]
            attn[j, :n] = b["attention_mask"]
            lbls[j, :n] = b["labels"]
        return {"input_ids": ids, "attention_mask": attn, "labels": lbls}

    def train_lora_layers(self, dataset, steps=200, lr=2e-4, batch_size=4,
                          mode="lora", seed=0, log_every=20):
        torch.manual_seed(seed)
        self.freeze_base()
        params = self.trainable(mode=mode)
        if not params: raise RuntimeError("nothing trainable; wrap() first")
        opt = AdamW(params, lr=lr, weight_decay=1e-4)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=self._collate, drop_last=False)
        self.model.train()
        it, last = 0, None
        while it < steps:
            for batch in dl:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                it += 1; last = loss.item()
                if log_every and it % log_every == 0:
                    print(f"step {it:4d} loss {last:.4f}")
                if it >= steps: break
        self.model.eval()
        return last

    # ---- persistence ----
    def save_lora_weights(self, path):
        state = {}
        for L, w in self.tweaks.items():
            state[f"L{L}"] = {"A": w.A.detach().cpu(), "B": w.B.detach().cpu(),
                              "bias": w.bias.detach().cpu(),
                              "r": w.r, "alpha": w.alpha, "lora_scale": w.lora_scale}
        torch.save(state, path)

    def load_lora_weights(self, path):
        state = torch.load(path, map_location="cpu")
        for key, s in state.items():
            L = int(key[1:])
            if L not in self.tweaks:
                self.wrap(L, r=s["r"], alpha=s["alpha"])
            w = self.tweaks[L]
            w.A.data.copy_(s["A"].to(w.A.device)); w.B.data.copy_(s["B"].to(w.B.device))
            w.bias.data.copy_(s["bias"].to(w.bias.device))
            w.lora_scale = s["lora_scale"]
        return self

    # ---- chat ----
    def continue_chat(self, session, prompt, max_new_tokens=256, stream=True, **gen_kw):
        session.append({"role": "user", "content": prompt})
        ids = self.tok.apply_chat_template(
            session, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False).to(self.model.device)
        streamer = TextStreamer(self.tok, skip_prompt=True) if stream else None
        out = self.model.generate(**ids, streamer=streamer,
                                  pad_token_id=self.tok.eos_token_id,
                                  max_new_tokens=max_new_tokens,
                                  repetition_penalty=1.25, **gen_kw)
        new = out[0, ids["input_ids"].shape[1]:]
        session.append({"role": "assistant",
                        "content": self.tok.decode(new, skip_special_tokens=False)})
        return session

    # ---- cleanup ----
    def clear(self):
        for L, w in list(self.tweaks.items()):
            self.layers[L].mlp.down_proj = w.base
        self.tweaks.clear()
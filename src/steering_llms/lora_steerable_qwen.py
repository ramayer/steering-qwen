import json
import torch, torch.nn as nn
import time
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from huggingface_hub import snapshot_download

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
        return out + lora.to(out.dtype) + self.bias.to(out.dtype) * self.lora_scale

    def trainable(self, mode="lora"):
        if mode == "lora": return [self.A, self.B]
        if mode == "bias": return [self.bias]
        if mode == "both": return [self.A, self.B, self.bias]
        raise ValueError(mode)


# ---------- dataset ----------

class ChatDataset(Dataset):
    """conversations: list of [[role, text], ...]. Trains only on 'assistant' turns.
    Items whose assistant span is entirely truncated away are dropped at construction."""
    def __init__(self, conversations, tok, max_len=256, verbose=True):
        self.tok, self.max_len = tok, max_len
        self.items = []
        dropped = 0
        for conv in conversations:
            item = self._build(conv)
            if item is None:
                dropped += 1
            else:
                self.items.append(item)
        if verbose and dropped:
            print(f"ChatDataset: kept {len(self.items)}, dropped {dropped} "
                  f"(assistant span lost to max_len={max_len})")

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

    def _build(self, conv):
        turns = [{"role": r, "content": t} for r, t in conv]
        full = self.tok.apply_chat_template(
            turns, add_generation_prompt=False, return_tensors="pt",
            return_dict=True, enable_thinking=False)
        ids = full["input_ids"][0][:self.max_len]
        labels = torch.full_like(ids, -100)
        for k, (role, text) in enumerate(conv):
            if role != "assistant": continue
            prefix = self.tok.apply_chat_template(
                turns[:k], add_generation_prompt=True, return_tensors="pt",
                return_dict=True, enable_thinking=False)["input_ids"][0]
            seg = self.tok(text, add_special_tokens=False)["input_ids"]
            start = len(prefix)
            end = min(start + len(seg), self.max_len)
            labels[start:end] = ids[start:end]
        if (labels != -100).sum() == 0:
            return None
        return {"input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "labels": labels}
        
# ---------- the model ----------

class SteeredModel:
    def __init__(self, model_id="Qwen/Qwen3.5-4B", calib_prompts=None,
                 max_memory=None, dtype=torch.bfloat16):
        
        # Regarding cached files:
        # https://share.google/aimode/fk6Osql9l52aauOlG
        local_dir = snapshot_download(repo_id=model_id)
        self.tok = AutoTokenizer.from_pretrained(local_dir,local_files_only=True)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            local_dir, quantization_config=q, device_map="auto",
            max_memory=max_memory or {0: "5GiB", "cpu": "16GiB"},
            local_files_only=True,
            dtype=dtype)
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
        
    def train_lora_layers(self, dataset, opt_steps=200, lr=2e-4,
                          micro_batch_size=1, grad_accum=4, mode="lora",
                          seed=42, log_every=20, max_grad_norm=1.0):

        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False,  "every_n_layers": 1}
        )
        torch.manual_seed(seed)
        self.freeze_base()
        params = self.trainable(mode=mode)
        if not params:
            raise RuntimeError("nothing trainable; wrap() first")
        opt = AdamW(params, lr=lr, weight_decay=1e-4)
        macro_batch = micro_batch_size * grad_accum
        dl = DataLoader(dataset, batch_size=macro_batch, shuffle=True,
                        collate_fn=self._collate, drop_last=True)
        self.model.train()
        opt_step, last_loss, t0 = 0, None, time.time()    
        while opt_step < opt_steps:
            for macro in dl:
                opt.zero_grad()
                accum = 0.0
                for i in range(0, macro["input_ids"].size(0), micro_batch_size):
                    micro = {k: v[i:i+micro_batch_size].to(self.model.device)
                             for k, v in macro.items()}
                    loss = self.model(**micro).loss / grad_accum
                    loss.backward()
                    accum += loss.item()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                opt.step()
                opt_step += 1
                last_loss = accum
                if log_every and opt_step % log_every == 0:
                    print(f"opt_step {opt_step:4d}  loss {last_loss:.4f} at {time.time() - t0:.2f}")
                    self.model.eval()
                    sess = self.continue_chat([],"What is 5+5?", max_new_tokens=50)
                    self.model.train()
                if opt_step >= opt_steps:
                    break
        self.model.eval()
        return {"opt_steps": opt_step, "last_loss": last_loss}
        
        
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
    def continue_chat(self, session, prompt, 
                      max_new_tokens=256, stream=True, 
                      repetition_penalty=None,
                      add_generation_prompt=True, enable_thinking=False,
                      continue_final_message=False, debug_input=False,
                      **gen_kw):
        session = json.loads(json.dumps(session))
        if prompt is not None:
            session.append({"role": "user", "content": prompt})
        else:
            add_generation_prompt=False
            continue_final_message=True
            enable_thinking=False

        input_txt = self.tok.apply_chat_template(
            session,
            tokenize=False,             
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=enable_thinking
        )
        ids = self.tok(input_txt, return_tensors="pt", return_dict=True).to(self.model.device)
        if debug_input:
            print(input_txt)
        streamer = TextStreamer(self.tok, skip_prompt=True, skip_special_tokens=False) if stream else None
        outputs = self.model.generate(**ids, 
                                  streamer=streamer,
                                  pad_token_id=self.tok.eos_token_id,
                                  max_new_tokens=max_new_tokens,
                                  repetition_penalty=repetition_penalty, 
                                  **gen_kw
                                  )
        input_len = ids["input_ids"].shape[1]
        new_tokens = outputs[0, input_len:]
        if new_tokens[-1] in self.tok.all_special_ids and len(new_tokens)<=2:
            print("### End reached",  self.tok.decode(new_tokens, skip_special_tokens=False))
        new_text = self.tok.decode(new_tokens, skip_special_tokens=True)
        if session[-1]['role'] == "assistant":
            session[-1]["content"] += new_text
        else:
            session = session + [{"role": "assistant","content": new_text}]
        return session

    # ---- cleanup ----
    def clear(self):
        for L, w in list(self.tweaks.items()):
            self.layers[L].mlp.down_proj = w.base
        self.tweaks.clear()

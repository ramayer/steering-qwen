import torch, torch.nn as nn
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

class NeuronTweak:
    """Fixed hook-based offset at down_proj output. Dose = fraction of median residual norm."""
    def __init__(self, layer, neuron, strength, calib_norm, down_proj):
        self.layer, self.neuron, self.strength = layer, neuron, strength
        self.calib_norm = calib_norm
        with torch.no_grad():
            e = torch.zeros(1, down_proj.in_features, 
                            dtype=torch.bfloat16, # not down_proj.weight.dtype,
                            device=down_proj.weight.device)
            e[0, neuron] = 1.0
            col = down_proj(e)[0].float()
        self._vec = (col / col.norm() * strength * calib_norm).to(torch.bfloat16)
        self._handle = None

    @property
    def dose_in_med_norms(self):
        return float(self._vec.norm() / self.calib_norm)

    def attach(self, down_proj):
        def hook(mod, inp, out):
            return out + self._vec.to(device=out.device, dtype=out.dtype)
        self._handle = down_proj.register_forward_hook(hook)

    def detach(self):
        if self._handle is not None:
            self._handle.remove(); self._handle = None


class LoRATweak(nn.Module):
    """
      Wraps down_proj: y = Wx + (B@A)x * lora_scale + bias.

      dp.weight.dtype on a 4-bit Params4bit is torch.uint8 (the packed storage), 
      so bias.to(uint8) gives an integer tensor, and nn.Parameter on an integer 
      tensor can't require grad. That's the whole error.

      The fix: the bias needs to be a floating-point tensor, matching the 
      compute dtype of the layer, not the storage dtype. For bnb 4-bit, 
      that's bnb_4bit_compute_dtype (you set it to bfloat16). 
      Same for A/B — they should be float, not the quantized type.
    """

    def __init__(self, base, bias, r=8, alpha=16, compute_dtype=torch.bfloat16):
        super().__init__()
        self.base = base
        self.r, self.alpha = r, alpha
        self.lora_scale = alpha / r
        d_out, d_in = base.out_features, base.in_features
        dev = base.weight.device          # follow the 4-bit weight's device
        self.A = nn.Parameter(torch.randn(r, d_in, dtype=compute_dtype, device=dev) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r, dtype=compute_dtype, device=dev))
        self.bias = nn.Parameter(bias.to(device=dev, dtype=compute_dtype).clone())

    def forward(self, x):
        out = self.base(x)
        A, B, b = self.A.to(x.device), self.B.to(x.device), self.bias.to(x.device)
        lora = ((x.to(A.dtype) @ A.T) @ B.T) * self.lora_scale
        return out + lora.to(out.dtype) + b.to(out.dtype)
            
    def trainable(self, mode="lora"):        # "lora" | "bias" | "both"
        if mode == "lora": return [self.A, self.B]
        if mode == "bias": return [self.bias]
        return [self.A, self.B, self.bias]

class SteeredModel:
    def __init__(self, model_id="Qwen/Qwen3.5-4B", calib_prompts=None,
                 max_memory=None, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        q = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=q, device_map="auto",
            max_memory=max_memory or {0: "5GiB", "cpu": "16GiB"}, 
            dtype=dtype
        )
        self.layers, self.layer_path = self._find_layers()
        self.tweaks = {}                 # layer -> [NeuronTweak | LoRATweak]
        self._med = {}
        self._calib = calib_prompts or [
            "The quick brown fox jumps over the lazy dog.",
            "Explain photosynthesis in one sentence.",
            "What is the capital of France?",
            "Write a haiku about rain.",
        ]

    # ---- layout discovery ----
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
    def median_norm(self, layer, exclude_first_token=True):
        """Median ||residual leaving layer| over calib prompts. Cached."""
        if layer in self._med: return self._med[layer]
        norms = []
        for t in self._calib:
            enc = self.tok(t, return_tensors="pt").to(self.model.device)
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            h = hs[layer + 1][0]
            if exclude_first_token: h = h[1:]
            norms.append(h.float().norm(dim=-1))
        self._med[layer] = torch.cat(norms).median().item()
        return self._med[layer]

    def _calibrate_all(self, layers):
        for L in set(layers): self.median_norm(L)

    def _check_unsteered(self):
        if any(any(isinstance(t, LoRATweak) for t in ts) for ts in self.tweaks.values()):
            raise RuntimeError("calibrate before injecting tweaks, or zero biases first")

    # ---- (a) fixed neuron tweak ----
    def tweak(self, layer, neuron, strength=1.0):
        """
            Add a fixed neuron offset. 
            strength=1.0 == one median residual norm at that layer.
        """
        self._check_unsteered()
        calib = self.median_norm(layer)
        dp = self.layers[layer].mlp.down_proj
        if isinstance(dp, LoRATweak):
            raise RuntimeError(f"layer {layer} already wrapped as LoRATweak; use .wrap()")
        t = NeuronTweak(layer, neuron, strength, calib, dp)
        t.attach(dp)
        self.tweaks.setdefault(layer, []).append(t)
        return t

    # ---- (a') multi-neuron wrap (pre-init bias, trainable LoRA) ----
    def wrap(self, config, r=8, alpha=16):
        """config = [(layer, neuron, strength), ...]. Returns {layer: LoRATweak}."""
        self._calibrate_all(L for L, _, _ in config)
        by_layer = {}
        for L, n, s in config: by_layer.setdefault(L, []).append((n, s))
        out = {}
        for L, items in by_layer.items():
            dp = self.layers[L].mlp.down_proj
            if isinstance(dp, LoRATweak):
                raise RuntimeError(f"layer {L} already wrapped")
            calib = self._med[L]
            with torch.no_grad():
                bias = torch.zeros(dp.out_features, dtype=torch.float32, device=dp.weight.device)
                for n, s in items:
                    e = torch.zeros(1, dp.in_features, dtype=torch.bfloat16, device=dp.weight.device)
                    e[0, n] = 1.0
                    col = dp(e)[0].float()
                    bias += (s * calib) * col / col.norm()
            w = LoRATweak(dp, bias, r=r, alpha=alpha)
            self.layers[L].mlp.down_proj = w
            self.tweaks.setdefault(L, []).append(w)
            out[L] = w
        return out

    # ---- (b) training ----
    def trainable_params(self):
        ps = []
        for ts in self.tweaks.values():
            for t in ts:
                if isinstance(t, LoRATweak): ps += t.trainable()
        return ps

    def freeze_base(self):
        for p in self.model.parameters(): p.requires_grad_(False)
        for ts in self.tweaks.values():
            for t in ts:
                if isinstance(t, LoRATweak):
                    t.A.requires_grad_(True); t.B.requires_grad_(True)

    def train_distill(self, texts, steps=200, lr=2e-4, T=2.0, max_len=128):
        """Distill teacher (bias only, lora_scale=0) into student (lora_scale restored)."""
        self.freeze_base()
        params = self.trainable_params()
        if not params: raise RuntimeError("nothing trainable; call .wrap() first")
        opt = AdamW(params, lr=lr)
        wrappers = [t for ts in self.tweaks.values() for t in ts if isinstance(t, LoRATweak)]
        real_scale = {id(w): w.lora_scale for w in wrappers}

        def set_teacher(on):
            for w in wrappers: w.lora_scale = 0.0 if on else real_scale[id(w)]

        self.model.train()
        for step in range(steps):
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len).to(self.model.device)
            with torch.no_grad():
                set_teacher(True)
                t_logits = self.model(**enc).logits
            set_teacher(False)
            s_logits = self.model(**enc).logits
            s_logp = F.log_softmax(s_logits / T, dim=-1)
            t_p = F.softmax(t_logits / T, dim=-1)
            kl = F.kl_div(s_logp, t_p, reduction="none").sum(-1)
            loss = (kl * enc.attention_mask).sum() / enc.attention_mask.sum() * T * T
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 20 == 0: print(f"step {step:4d}  KL {loss.item():.4f}")
        self.model.eval()
        return wrappers

    # ---- (c) chat ----
    def continue_chat(self, session, prompt, max_new_tokens=256, stream=True, 
                      repetition_penalty=None,**gen_kw):
        session.append({"role": "user", "content": prompt})
        ids = self.tok.apply_chat_template(
            session, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False).to(self.model.device)
        streamer = TextStreamer(self.tok, skip_prompt=True) if stream else None
        out = self.model.generate(**ids, streamer=streamer,
                                  pad_token_id=self.tok.eos_token_id,
                                  max_new_tokens=max_new_tokens,
                                  repetition_penalty=repetition_penalty, 
                                  **gen_kw)
        new = out[0, ids["input_ids"].shape[1]:]
        session.append({"role": "assistant",
                        "content": self.tok.decode(new, skip_special_tokens=False)})
        return session

    # ---- cleanup ----
    def clear(self):
        for ts in self.tweaks.values():
            for t in ts:
                if isinstance(t, NeuronTweak): t.detach()
        self.tweaks.clear()


    def clear_all(self):
        for L, ts in list(self.tweaks.items()):
            for t in ts:
                if isinstance(t, NeuronTweak):
                    t.detach()
                elif isinstance(t, LoRATweak):
                    # put the original down_proj back
                    self.layers[L].mlp.down_proj = t.base
        self.tweaks.clear()
        self._med.clear()          # optional: drop cached calibration too

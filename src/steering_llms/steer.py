from __future__ import annotations

"""Neuron steering: add a scaled ``down_proj`` column to a layer's MLP output via a forward hook.

Two ways to size the push, both from the original notebook:

* ``raw``  -- ``amount * c / ||c||``: a unit direction times a hand-picked amount
  (e.g. layer 14 neuron 8949 at 7.5 is "evil", layer 17 neuron 7426 at 7.5 is "poetic").
* ``dose`` -- ``dose * median_norm(L) * c / ||c||``: dose 1.0 pushes as hard as the length of a
  typical residual vector leaving layer ``L``, so doses are comparable across layers.
"""
import contextlib
import sys
import json
import time
    
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextStreamer

from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .generate import chat, chat_inputs
from .model import find_layers

Spec = list[tuple[int, int, float]]  # [(layer, neuron, amount_or_dose), ...]
Mode = Literal["raw", "dose"]
from .lora import LoRATweak
from .model import load_qwen, load_model, find_decoder_layers

if TYPE_CHECKING:
    from .data import ChatDataset

# ---------- module-level helpers (no SteeredModel coupling) ----------
    
class SteeredModel:
    """Wraps a (possibly 4-bit) causal LM and lets you steer it at chosen MLP layers.

    Core state:
        self.model   the HF model (kept public; nothing is hidden)
        self.layers  the decoder layer list, found at init
        self.tweaks  {layer: LoRATweak}, one per wrapped layer
    """

    def __init__(
            self,
            model_id: str = "Qwen/Qwen3.5-4B",
            calib_prompts: list[str] | None = None,
            dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.tok, self.model = load_model(model_id,verbose=True)
        self.layers, self.layer_path = find_decoder_layers(self.model)
        self.dtype = dtype
        self.tweaks: dict[int, LoRATweak] = {}
        self._base_med: dict[int, float] = {}
        self._calib = calib_prompts or [
            "The quick brown fox jumps over the lazy dog.",
            "Explain photosynthesis in one sentence.",
            "What is the capital of France?",
            "Write a haiku about rain.",
        ]
        if list(self.model.parameters())[0].device.type == 'cuda':
            print("enabling gradient checkpointing (reentrant=False, every_n_layers=1) for memory savings")
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "every_n_layers": 1}
            )

    # ---- calibration --------------------------------------------------------

    @contextlib.contextmanager
    def _base_view(self):
        """Temporarily swap every LoRATweak back to its base module."""
        saved = {}
        for L, w in self.tweaks.items():
            self.layers[L].mlp.down_proj = w.base
            saved[L] = w
        try:
            yield
        finally:
            for L, w in saved.items():
                self.layers[L].mlp.down_proj = w

    @torch.no_grad()
    def _fill_base_med(self):
        """One calibration pass fills base_median_norm for *every* layer at once."""
        if self._base_med:
            return
        all_norms = {l: [] for l in range(len(self.layers) + 1)}
        with self._base_view():
            for t in self._calib:
                enc = self.tok(t, return_tensors="pt").to(self.model.device)
                hs = self.model(**enc, output_hidden_states=True).hidden_states
                for l in range(len(self.layers) + 1):
                    h = hs[l][0, 1:].float()
                    all_norms[l].append(h.norm(dim=-1))
        for l, ns in all_norms.items():
            self._base_med[l] = torch.cat(ns).median().item()

    def base_median_norm(self, L):
        """Median ||residual leaving layer L|| for the *base* model. Cached for all L."""
        if L not in self._base_med:
            self._fill_base_med()
        return self._base_med[L]

    def median_norm(self, L):
        """Alias: by default, dose units are defined against the base model."""
        return self.base_median_norm(L)

    # ---- neuron columns -----------------------------------------------------

    def _check(self, L, n):
        if not 0 <= L < len(self.layers):
            raise ValueError(f"layer {L} out of range (model has {len(self.layers)} layers)")
        width = self.layers[L].mlp.down_proj.in_features
        if not 0 <= n < width:
            raise ValueError(f"neuron {n} out of range (layer {L} MLP has {width} neurons)")

    @torch.no_grad()
    def down_col(self, L, n):
        """Column n of layer L's *effective* down_proj, float32 [d_model].

        Uses __call__ (not .forward) so active steering on this layer is reflected --
        the current model is the real model. To measure the base substrate, use
        _base_view() or clear() first.
        """
        self._check(L, n)
        dp = self.layers[L].mlp.down_proj
        dev = dp.weight.device if dp.weight.device.type != "meta" else self.model.device
        e = torch.zeros(1, dp.in_features, dtype=self.model.dtype, device=dev)
        e[0, n] = 1.0
        return dp(e)[0].float()

    @torch.no_grad()
    def neuron_vector(self, L, n, calibrated=True):
        """Unit-normalized column n, optionally scaled by median residual norm at L."""
        col = self.down_col(L, n)
        v = col / col.norm()
        return v * self.median_norm(L) if calibrated else v

    # ---- wrapping -----------------------------------------------------------

    def wrap(self, *layers, r=8, alpha=16, bias=None, dtype=torch.bfloat16):
        """Wrap each layer's down_proj with a LoRATweak (default bias = zeros).
        Returns {layer: LoRATweak}; also stored in self.tweaks."""
        out = {}
        for L in layers:
            if L in self.tweaks:
                raise RuntimeError(f"layer {L} already wrapped")
            dp = self.layers[L].mlp.down_proj
            w = LoRATweak(dp, bias=bias, r=r, alpha=alpha, compute_dtype=dtype)
            self.layers[L].mlp.down_proj = w
            self.tweaks[L] = w
            out[L] = w
        return out

    # ---- fixed steering -----------------------------------------------------

    def set_neuron_bias(self, L, n, amount=1.0, calibrated=True, verbose=False):
        """Add a neuron's steering vector to layer L's bias. Wraps silently if needed.
        Multiple calls on the same layer accumulate, so two neurons compose."""
        if L not in self.tweaks:
            self.wrap(L)
        v = self.neuron_vector(L, n, calibrated=calibrated) * amount
        if verbose:
            print(f"L{L} n{n}: ||col||={self.down_col(L, n).norm():.3f}  "
                  f"median_norm={self.median_norm(L):.3f}  ||add||={v.norm():.3f}")
        self.tweaks[L].bias.data.add_(v.to(self.tweaks[L].bias.dtype))

    def zero_bias(self, L):
        """Reset layer L's bias to zero (undo all set_neuron_bias calls on that layer)."""
        if L in self.tweaks:
            self.tweaks[L].bias.data.zero_()

    @contextlib.contextmanager
    def steering(self, spec, calibrated=True, verbose=False):
        """with sm.steering([(14, 8949, 1.0), (17, 7426, 0.75)]): ...

        Additive and non-destructive: the added bias is snapshotted and restored on
        exit, and layers wrapped solely for this block are unwrapped again (if the
        wrapper is back to a no-op).
        """
        snapshots = {}          # L -> saved bias tensor
        wrapped_here = []       # layers we wrapped for this block
        for L, n, amount in spec:
            if L not in self.tweaks:
                self.wrap(L)
                wrapped_here.append(L)
            if L not in snapshots:
                snapshots[L] = self.tweaks[L].bias.detach().clone()
            v = self.neuron_vector(L, n, calibrated=calibrated) * amount
            if verbose:
                print(f"L{L} n{n}: ||add||={v.norm():.3f}")
            self.tweaks[L].bias.data.add_(v.to(self.tweaks[L].bias.dtype))
        try:
            yield self
        finally:
            for L, snap in snapshots.items():
                self.tweaks[L].bias.data.copy_(snap)
            for L in wrapped_here:
                w = self.tweaks[L]
                if (w.bias == 0).all() and (w.B == 0).all():
                    self.layers[L].mlp.down_proj = w.base
                    del self.tweaks[L]

    # ---- training -----------------------------------------------------------

    def freeze_base(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def trainable(self, mode="lora"):
        """Collect and enable the trainable params of every wrapper in self.tweaks."""
        ps = []
        for w in self.tweaks.values():
            ps += w.trainable(mode)
        for p in ps:
            p.requires_grad_(True)
        return ps

    # def _collate(self, batch):
    #     pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
    #     maxlen = max(b["input_ids"].size(0) for b in batch)
    #     ids  = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    #     attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
    #     lbls = torch.full((len(batch), maxlen), -100, dtype=torch.long)
    #     for j, b in enumerate(batch):
    #         n = b["input_ids"].size(0)
    #         ids[j, :n]  = b["input_ids"]
    #         attn[j, :n] = b["attention_mask"]
    #         lbls[j, :n] = b["labels"]
    #     return {"input_ids": ids, "attention_mask": attn, "labels": lbls}

    def train_lora_layers(
        self,
        dataset: "ChatDataset",
        steps: int = 200,
        lr: float = 2e-4,
        micro_batch_size: int = 1,
        macro_batch_size: int = 4,
        mode: str = "lora",
        seed: int = 42,
        max_grad_norm: float = 1.0,
        debug_interval: float = 60,
        debug_prompt: str = "How can I bake a pizza?",
        collate_fn: Callable | None = None,
    ) -> dict[str, float | int | None]:
        """Train the adapter parameters on ``dataset`` with gradient accumulation.

        The dataset must yield dicts with ``input_ids``, ``attention_mask``, and
        ``labels`` (``-100`` outside assistant spans). ``collate_fn`` defaults to
        ``dataset.collate`` if present.

        Returns:
            ``{"opt_steps": int, "last_loss": float | None}``.
        """
        torch.manual_seed(seed)
        self.freeze_base()
        params = self.trainable(mode=mode)
        if not params:
            raise RuntimeError("nothing trainable; wrap() first")
        opt = AdamW(params, lr=lr, weight_decay=1e-4)
        grad_accum = macro_batch_size // micro_batch_size
        assert len(dataset) >= 1, "empty dataset after filtering"
        collate_fn = collate_fn or getattr(dataset, "collate", None)
        dl = DataLoader(
            dataset,
            batch_size=macro_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        )
        self.model.train()
        step, last_loss, t0 = 0, None, time.time()
        next_debug_time = t0 + debug_interval
        print("trainng loop beginning")
        with tqdm(total=steps, desc="Training") as pbar:
            while step < steps:
                for macro in dl:
                    opt.zero_grad()
                    accum = 0.0
                    batch_count = 0
                    for i in range(0, macro["input_ids"].size(0), micro_batch_size):
                        micro = {
                            k: v[i : i + micro_batch_size].to(self.model.device)
                            for k, v in macro.items()
                        }
                        loss = self.model(**micro).loss / grad_accum
                        loss.backward()
                        accum += loss.item()
                        step += micro_batch_size
                        batch_count += micro_batch_size
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                    opt.step()
                    last_loss = accum
                    pbar.update(batch_count)
                    pbar.set_postfix(loss=f"{last_loss:.4f}") # Optional: display live metrics
                    if time.time() > next_debug_time:
                        self.model.eval()
                        self.continue_chat([], debug_prompt, max_new_tokens=79, stream=True)
                        self.model.train()
                        debug_interval *= 1.5
                        next_debug_time = time.time() + debug_interval
                    if step >= steps:
                        break
        self.model.eval()
        return {"steps": steps, "last_loss": last_loss}
    # ---- persistence --------------------------------------------------------

    def save_lora_weights(self, path):
        state = {}
        for L, w in self.tweaks.items():
            state[f"L{L}"] = {
                "A": w.A.detach().cpu(),
                "B": w.B.detach().cpu(),
                "bias": w.bias.detach().cpu(),
                "r": w.r, "alpha": w.alpha, "lora_scale": w.lora_scale,
            }
        torch.save(state, path)

    def load_lora_weights(self, path):
        state = torch.load(path, map_location="cpu")
        for key, s in state.items():
            L = int(key[1:])
            if L not in self.tweaks:
                self.wrap(L, r=s["r"], alpha=s["alpha"])
            w = self.tweaks[L]
            w.A.data.copy_(s["A"].to(w.A.device))
            w.B.data.copy_(s["B"].to(w.B.device))
            w.bias.data.copy_(s["bias"].to(w.bias.device))
            w.lora_scale = s["lora_scale"]
        return self

    # ---- chat ---------------------------------------------------------------


    # ---- chat ----
    def continue_chat(
        self,
        session: list[dict[str, str]],
        prompt: str | None,
        max_new_tokens: int = 256,
        stream: bool = True,
        repetition_penalty: float | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        continue_final_message: bool = False,
        debug_input: bool = False,
        **gen_kw,
    ) -> list[dict[str, str]]:
        """Append ``prompt`` as a user turn, generate, and append the reply.

        ``prompt=None`` means "continue the last assistant turn" and requires a
        non-empty session ending in an assistant message; otherwise an
        ``IndexError`` is raised by the template renderer.

        Returns:
            The updated session (a new list; the input is not mutated).
        """
        session = json.loads(json.dumps(session))
        if prompt is not None:
            session.append({"role": "user", "content": prompt})
        else:
            add_generation_prompt = False
            continue_final_message = True
            enable_thinking = False

        input_txt = self.tok.apply_chat_template(
            session,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=enable_thinking,
        )
        ids = self.tok(input_txt, return_tensors="pt", return_dict=True).to(
            self.model.device
        )
        if debug_input:
            print(input_txt)
        streamer = (
            TextStreamer(self.tok, skip_prompt=True, skip_special_tokens=False)
            if stream
            else None
        )
        outputs = self.model.generate(
            **ids,
            streamer=streamer,
            pad_token_id=self.tok.eos_token_id,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            **gen_kw,
        )
        input_len = ids["input_ids"].shape[1]
        new_tokens = outputs[0, input_len:]
        if (
            len(new_tokens) >= 1
            and new_tokens[-1] in self.tok.all_special_ids
            and len(new_tokens) <= 2
        ):
            print(
                "### End reached",
                self.tok.decode(new_tokens, skip_special_tokens=False),
            )
        new_text = self.tok.decode(new_tokens, skip_special_tokens=True)
        if session and session[-1]["role"] == "assistant":
            session[-1]["content"] += new_text
        else:
            session = session + [{"role": "assistant", "content": new_text}]
        return session


    # ---- cleanup ------------------------------------------------------------

    def clear(self):
        """Unwrap every layer, restoring the original down_proj. Base-norm cache kept."""
        for L, w in list(self.tweaks.items()):
            self.layers[L].mlp.down_proj = w.base
        self.tweaks.clear()

#######################################################################
# deprecated
#######################################################################

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .generate import chat, chat_inputs
from .model import find_layers

Spec = list[tuple[int, int, float]]  # [(layer, neuron, amount_or_dose), ...]
Mode = Literal["raw", "dose"]

CALIB: list[str] = [
    "Tell me about your weekend.",
    "How do I bake bread?",
    "Explain photosynthesis briefly.",
    "What's a good name for a cat?",
]


class Steerer:
    def __init__(
        self,
        model: PreTrainedModel,
        tok: PreTrainedTokenizerBase,
        calib: list[str] | None = None,
    ) -> None:
        self.model = model
        self.tok = tok
        self.calib = calib if calib is not None else CALIB
        self.layers, self.layers_path = find_layers(model)
        self._handles: list = []
        self._active: dict[int, Tensor] = {}
        self._med: dict[int, float] = {}

    # ---- hooks -------------------------------------------------------------------------------

    #done
    def clear(self) -> None:
        """Remove all steering hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._active.clear()

    def _add(self, L: int, v: Tensor) -> None:
        def hook(mod, inp, out):
            # .to() is a no-op once v matches; kept lazy so accelerate-offloaded layers work too
            return out + v.to(device=out.device, dtype=out.dtype)

        self._handles.append(self.layers[L].mlp.down_proj.register_forward_hook(hook))
        self._active[L] = v

    @property
    def active(self) -> bool:
        return bool(self._handles)

    # ---- directions and scales ---------------------------------------------------------------
    #done
    def _check(self, L: int, n: int) -> None:
        if not 0 <= L < len(self.layers):
            raise ValueError(f"layer {L} out of range (model has {len(self.layers)} layers)")
        width = self.layers[L].mlp.down_proj.in_features
        if not 0 <= n < width:
            raise ValueError(f"neuron {n} out of range (layer {L} MLP has {width} neurons)")

    # done
    @torch.no_grad()
    def down_col(self, L: int, n: int) -> Tensor:
        """Column ``n`` of layer ``L``'s ``down_proj`` as float32 ``[d_model]``.

        Computed by pushing a one-hot through the module so it also works for quantized
        (bitsandbytes) weights, where the column isn't directly readable. Calls ``.forward``
        directly so active steering hooks on this layer don't leak into the result.
        """
        self._check(L, n)
        dp = self.layers[L].mlp.down_proj
        device = dp.weight.device if dp.weight.device.type != "meta" else self.model.device  # meta = offloaded
        e = torch.zeros(1, dp.in_features, dtype=self.model.dtype, device=device)
        e[0, n] = 1.0
        return dp.forward(e)[0].float()

    # done
    @torch.no_grad()
    def median_norm(self, L: int) -> float:
        """Median ||residual leaving layer L|| over the calibration prompts (excluding position 0).

        Always measured on the unsteered model: active hooks are lifted while measuring.
        """
        if L in self._med:
            return self._med[L]
        saved = dict(self._active)
        self.clear()
        try:
            norms: list[Tensor] = []
            for t in self.calib:
                enc = chat_inputs(self.tok, t, self.model.device)
                hs = self.model(**enc, output_hidden_states=True).hidden_states  # (n_layers+1) x [1, seq, d]
                norms.append(hs[L + 1][0, 1:].float().norm(dim=-1))
            self._med[L] = torch.cat(norms).median().item()
        finally:
            for layer, v in saved.items():
                self._add(layer, v)
        return self._med[L]

    def vectors(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> dict[int, Tensor]:
        """Sum the steering vectors for ``spec`` per layer (float32)."""
        add: dict[int, Tensor] = {}
        for L, n, x in spec:
            c = self.down_col(L, n)
            scale = x / c.norm() if mode == "raw" else x * self.median_norm(L) / c.norm()
            v = scale * c
            if verbose:
                print(
                    f"  L{L} n{n} {mode}={x:+g}: ||col||={c.norm().item():.4f} ||push||={v.norm().item():.3f}",
                    file=sys.stderr,
                )
            add[L] = add[L] + v if L in add else v
        return add

    # ---- public steering API -----------------------------------------------------------------

    def set(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> None:
        """Replace any current steering with ``spec``."""
        self.clear()
        for L, v in self.vectors(spec, mode, verbose).items():
            self._add(L, v)

    def set_raw(self, spec: Spec, verbose: bool = False) -> None:
        self.set(spec, "raw", verbose)

    def set_dosed(self, spec: Spec, verbose: bool = False) -> None:
        self.set(spec, "dose", verbose)

    @contextmanager
    def steering(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> Generator["Steerer"]:
        """``with st.steering([(14, 8949, 7.5)]): ...`` -- hooks are removed on exit."""
        self.set(spec, mode, verbose)
        try:
            yield self
        finally:
            self.clear()

    def generate(self, prompt: str, max_new_tokens: int = 150, sample: bool = False, stream: bool = False) -> str:
        """Generate with whatever steering is currently active."""
        return chat(self.model, self.tok, prompt, max_new_tokens=max_new_tokens, sample=sample, stream=stream)

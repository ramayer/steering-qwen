from __future__ import annotations

"""
Multiple techniques for steering a causal LM at chosen MLP neurons.  The simplest is to add a fixed vector to the layer's bias, which is equivalent to adding a fixed vector to the layer's output.  The vector can be a column of the down_proj (the "weight" column) or a learned LoRA A@B term.

Two ways to size the push
* bias-based -- ``amount * c``: a fixed vector times a learned or hand-picked amount
* lora-based -- A@B term, where A is a readout matrix and B is a write matrix.  The A@B term is a conditional steer: it only fires when the inputs match the readout pattern in A.  The amount of push is scaled by lora_scale = alpha / r, where r is the rank of the LoRA matrices and alpha is a hyperparameter.
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

from .generate import ChatHelper, chat, chat_inputs
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
        model_id: str | None = "Qwen/Qwen3.5-4B",
        tok: PreTrainedTokenizerBase | None = None,
        model: PreTrainedModel | None = None,
        calib_prompts: list[str] | None = None,
        dtype: torch.dtype = torch.bfloat16,
        enable_gradient_checkpointing: bool | None = None,
    ) -> None:
        """Either pass model_id (loads from disk) or pass both tok and model
        (wraps an already-loaded pair). Passing both model_id and (tok, model)
        is an error."""
        if model is not None or tok is not None:
            if model_id is not None:
                raise ValueError("pass either model_id or (tok, model), not both")
            if model is None or tok is None:
                raise ValueError("pass both tok and model")
        else:
            tok, model = load_model(model_id, verbose=True)

        self.tok: PreTrainedTokenizerBase = tok
        self.model: PreTrainedModel = model
        self.layers, self.layer_path = find_decoder_layers(self.model)
        self.dtype: torch.dtype = dtype
        self.tweaks: dict[int, LoRATweak] = {}
        self._base_med: dict[int, float] = {}
        self._calib: list[str] = calib_prompts or [
            "The quick brown fox jumps over the lazy dog.",
            "Explain photosynthesis in one sentence.",
            "What is the capital of France?",
            "Write a haiku about rain.",
        ]
        if enable_gradient_checkpointing is None:
            enable_gradient_checkpointing = next(self.model.parameters()).device.type == "cuda"
        if enable_gradient_checkpointing:
            print("enabling gradient checkpointing ...")
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "every_n_layers": 1})

    @classmethod
    def from_model(cls, model: PreTrainedModel, tok: PreTrainedTokenizerBase, **kwargs) -> "SteeredModel":
        """Build a SteeredModel around an already-loaded model and tokenizer."""
        return cls(model_id=None, tok=tok, model=model, **kwargs)

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
        assert isinstance(self.layers[L].mlp, nn.Module), f"layer {L} has no mlp"
        assert hasattr(self.layers[L].mlp, "down_proj"), f"layer {L} has no mlp.down_proj"
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

    def wrap(self, *layers, r=8, alpha=8, bias=None, dtype=torch.bfloat16):
        """Wrap each layer's down_proj with a LoRATweak (default bias = zeros).
        Returns {layer: LoRATweak}; also stored in self.tweaks."""
        out = {}
        for L in layers:
            if L in self.tweaks:
                raise RuntimeError(f"layer {L} already wrapped")
            dp = self.layers[L].mlp.down_proj # type: ignore
            w = LoRATweak(dp, bias=bias, r=r, alpha=alpha, compute_dtype=dtype)
            self.layers[L].mlp.down_proj = w # type: ignore
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
                print(f"in steering: L{L} n{n}: ||add||={v.norm():.3f}")
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
        next_debug_time = t0
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
                        self.continue_chat([], debug_prompt, max_new_tokens=40, stream=True)
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
            entry = {"r": w.r, "alpha": w.alpha, "lora_scale": w.lora_scale}
            if w.B.count_nonzero() > 0:          # branch is active -> save A and B together
                entry["A"] = w.A.detach().cpu()
                entry["B"] = w.B.detach().cpu()
            if w.bias.count_nonzero() > 0:       # unconditional steer is active
                entry["bias"] = w.bias.detach().cpu()
            state[f"L{L}"] = entry
            torch.save(state, path)

    def load_lora_weights(self, path):
        state = torch.load(path, map_location="cpu")
        for key, s in state.items():
            L = int(key[1:])
            if L not in self.tweaks:
                self.wrap(L, r=s["r"], alpha=s["alpha"])
            w = self.tweaks[L]
            # A and B travel together; if either is absent, the branch is zero
            if "A" in s and "B" in s:
                w.A.data.copy_(s["A"].to(w.A.device))
                w.B.data.copy_(s["B"].to(w.B.device))
            if "bias" in s:
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
        ch = ChatHelper(self.model, self.tok)
        return ch.continue_chat(
            session=session,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stream=stream,
            repetition_penalty=repetition_penalty,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            continue_final_message=continue_final_message,
            debug_input=debug_input,
            **gen_kw,
        )

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
    """Backward-compatible facade over SteeredModel for the cli tools."""
    def __init__(
        self,
        model: PreTrainedModel,
        tok: PreTrainedTokenizerBase,
        calib: list[str] | None = None,
    ) -> None:
        # build a SteeredModel around an already-loaded model
        sm = SteeredModel.__new__(SteeredModel)   # bypass __init__ / model load
        sm.tok, sm.model = tok, model
        sm.layers, sm.layer_path = find_decoder_layers(model)
        sm.dtype = next(model.parameters()).dtype
        sm.tweaks = {}
        sm._base_med = {}
        sm._calib = calib or CALIB
        self.sm = sm
        self._spec_snapshot = None   # for steering() restore

    # ---- attribute forwarding (old surface for tests) ----
    @property
    def model(self):  return self.sm.model
    @property
    def tok(self):    return self.sm.tok
    @property
    def layers(self): return self.sm.layers
    @property
    def calib(self):  return self.sm._calib
    @property
    def _med(self):   return self.sm._base_med

    # ---- hooks -------------------------------------------------------------------------------

    #done
    def clear(self) -> None:
        self.sm.clear()


    def _add(self, L: int, v: Tensor) -> None:
        raise NotImplementedError("hook based steering is deprecated; use SteeredModel.set() or .steering() instead")
        def hook(mod, inp, out):
            # .to() is a no-op once v matches; kept lazy so accelerate-offloaded layers work too
            return out + v.to(device=out.device, dtype=out.dtype)
        self._handles.append(self.layers[L].mlp.down_proj.register_forward_hook(hook))
        self._active[L] = v

    @property
    def active(self) -> bool:
        return any((w.bias != 0).any() or (w.B != 0).any()
                   for w in self.sm.tweaks.values())
    # ---- directions and scales ---------------------------------------------------------------
    #done
    def _check(self, L: int, n: int) -> None:
        return self.sm._check(L, n)

    # done
    @torch.no_grad()
    def down_col(self, L: int, n: int) -> Tensor:
        with self.sm._base_view():
            return self.sm.down_col(L, n)

    # done
    @torch.no_grad()
    def median_norm(self, L: int) -> float:
        with self.sm._base_view():
            return self.sm.median_norm(L)


    def vectors(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> dict[int, Tensor]:
        """Sum the steering vectors for ``spec`` per layer (float32)."""
        calibrated = (mode == "dose")
        add = {}
        for L, n, x in spec:
            v = self.sm.neuron_vector(L, n, calibrated=calibrated) * x
            add[L] = add[L] + v if L in add else v
            if verbose:
                col = self.down_col(L, n)
                v = col / col.norm()
                print(
                    f"  L{L} n{n} {mode}={x:+g}: " +
                    f"  ||col_norm||={col.norm().item():.4f} " +
                    f"  ||layer_median_norm||={self.median_norm(L):.3f}  "+
                    f"  ||push||={v.norm().item():.3f}",
                    file=sys.stderr,
                )
        return add

    # ---- public steering API -----------------------------------------------------------------

    def set(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> None:
        """Replace any current steering with ``spec``."""
        self.sm.clear()
        for L, v in self.vectors(spec, mode, verbose).items():
            if L not in self.sm.tweaks:
                self.sm.wrap(L)
            if verbose:
                print(f"in set L{L}: ||add||={v.norm():.3f}")
            self.sm.tweaks[L].bias.data.add_(v.to(self.sm.tweaks[L].bias.dtype))

    def set_raw(self, spec: Spec, verbose: bool = False) -> None:
        self.set(spec, "raw", verbose)

    def set_dosed(self, spec: Spec, verbose: bool = False) -> None:
        self.set(spec, "dose", verbose)

    @contextmanager
    def steering(self, spec: Spec, mode: Mode = "raw", verbose: bool = False) -> Generator["Steerer"]:
        calibrated = (mode == "dose")
        with self.sm.steering([(L, n, x) for L, n, x in spec], calibrated=calibrated, verbose=verbose):
            yield self

    def generate(self, prompt: str, max_new_tokens: int = 150, sample: bool = False, stream: bool = False) -> str:
        return self.sm.continue_chat([], prompt,
                                     max_new_tokens=max_new_tokens,
                                     stream=stream,
                                     do_sample=sample)[-1]["content"]
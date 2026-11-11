"""Shared fixtures: a tiny random Qwen3.5 (same architecture, ~16M params) on every available device.

The tiny model keeps Qwen3.5's hybrid layout (3 linear-attention layers + 1 full-attention layer)
and the real tokenizer, so tests exercise the same transformers code paths as the 4B model,
including bitsandbytes 4-bit (``Linear4bit``) loading, in a few seconds. Needs the Qwen3.5
config and tokenizer from the HF Hub (~12 MB, cached after first use).

Every device found is tested: cpu always, mps on Apple Silicon, cuda on an NVIDIA box.
Set REQUIRE_CUDA=1 to make the run fail if CUDA isn't visible, instead of quietly testing CPU only.
"""

import os

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from steering_llms import DEFAULT_MODEL, Steerer, load_model

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")
if torch.cuda.is_available():
    DEVICES.append("cuda")
elif os.environ.get("REQUIRE_CUDA"):
    raise pytest.UsageError("REQUIRE_CUDA is set but torch.cuda.is_available() is False")

# VARIANTS = [(d, q) for d in DEVICES for q in (False, True)]
#
# On a CUDA box, transformers picks the Triton (flash-linear-attention) kernels for Qwen3.5's
# linear-attention layers, and those can't run on CPU tensors.
_skip_cpu_on_cuda = pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="fla/Triton kernels can't run on CPU tensors",
)

VARIANTS = [
    pytest.param((d, q), marks=_skip_cpu_on_cuda if d == "cpu" else ())
    for d in DEVICES
    for q in (False, True)
]

def variant_id(v: tuple[str, bool]) -> str:
    return f"{v[0]}-{'nf4' if v[1] else 'bf16'}"


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> str:
    cfg = AutoConfig.from_pretrained(DEFAULT_MODEL).text_config
    cfg.update(
        dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            mtp_num_hidden_layers=0,
        )
    )
    # 64 * partial_rotary_factor(0.25) = 16 rotary dims = 8 frequency pairs
    cfg.rope_parameters = {**cfg.rope_parameters, "mrope_section": [3, 3, 2]}
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    path = tmp_path_factory.mktemp("tiny-qwen3_5")
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained(DEFAULT_MODEL).save_pretrained(path)
    return str(path)


_loaded: dict = {}


@pytest.fixture(params=VARIANTS, ids=variant_id)
def loaded(request, tiny_dir):
    """(tok, model, device, quantized), loaded once per variant per session."""
    device, quantize = request.param
    if request.param not in _loaded:
        tok, model = load_model(tiny_dir, device=device, quantize=quantize, verbose=False)
        _loaded[request.param] = (tok, model)
    tok, model = _loaded[request.param]
    return tok, model, device, quantize


@pytest.fixture
def st(loaded) -> Steerer:
    tok, model, _, _ = loaded
    s = Steerer(model, tok)
    yield s
    s.clear()

"""Device selection and model loading."""

import sys

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def pick_device(pref: str | None = None) -> torch.device:
    """Return ``pref`` if given, else the best available of cuda > mps > cpu."""
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def nf4_config() -> BitsAndBytesConfig:
    """The 4-bit config from the original notebook."""
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)


def load_model(
    model_id: str = DEFAULT_MODEL,
    device: str | torch.device | None = None,
    quantize: bool | None = None,
    max_memory: dict | None = None,
    verbose: bool = True,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load tokenizer and model in bf16, optionally 4-bit NF4 quantized via bitsandbytes.

    ``quantize`` defaults to True on CUDA (as in the original notebook) and False elsewhere.
    bitsandbytes also runs NF4 on MPS and CPU, so ``quantize=True`` on a Mac reproduces the
    CUDA run's weights; unquantized bf16 needs ~9 GB for the 4B model.

    ``max_memory`` only applies on CUDA, where the model is placed with ``device_map="auto"``;
    e.g. ``{0: "5GiB", "cpu": "16GiB"}`` for a small GPU.
    """
    dev = pick_device(str(device) if device is not None else None)
    if quantize is None:
        quantize = dev.type == "cuda"

    kwargs: dict = {"dtype": torch.bfloat16}
    if quantize:
        kwargs["quantization_config"] = nf4_config()
    if dev.type == "cuda":
        kwargs["device_map"] = "auto"
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
    else:
        kwargs["device_map"] = {"": str(dev)}

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    if verbose:
        gib = model.get_memory_footprint() / 2**30
        q = "4-bit NF4" if quantize else "unquantized"
        print(f"Loaded {model_id} on {model.device} ({model.dtype}, {q}, {gib:.2f} GiB)", file=sys.stderr)
    return tok, model


def find_layers(m: nn.Module) -> tuple[nn.ModuleList, str]:
    """Locate the decoder layer list (path differs between text-only and multimodal wrappers)."""
    for path in ["model.layers", "model.language_model.layers", "language_model.model.layers", "layers"]:
        obj = m
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and hasattr(obj[0], "mlp"):
            return obj, path
    raise RuntimeError("print(model) and locate the decoder layer list")

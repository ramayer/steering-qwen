"""Device selection and model loading."""
from __future__ import annotations

import gc
import os
import sys
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from warnings import deprecated # type: ignore

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
        os.environ['TRITON_DEFAULT_FP_FUSION']="0"
        kwargs["device_map"] = "auto"
        free, total = torch.cuda.mem_get_info()
        free_gb = int(free / 1024**3 - 0.25)
        if verbose:
            print(f"GPU Free: {free / 1024**2:.2f} MB, Total: {total / 1024**2:.2f} MB", file=sys.stderr)
        kwargs["max_memory"] = max_memory or {0: f"{free_gb}GiB", "cpu": "16GiB"}
    else:
        kwargs["device_map"] = {"": str(dev)}

    local_dir = snapshot_download(repo_id=model_id)

    tok = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(local_dir, **kwargs)
    model.eval()

    if verbose:
        gib = model.get_memory_footprint() / 2**30
        q = "4-bit NF4" if quantize else "unquantized"
        print(f"Loaded {model_id} on {model.device} ({model.dtype}, {q}, {gib:.2f} GiB)", file=sys.stderr)
    return tok, model
    
def find_decoder_layers(model: PreTrainedModel) -> tuple[nn.ModuleList, str]:
    """Locate the ``nn.ModuleList`` of decoder blocks in a HF causal LM.

    Tries the common attribute paths for Qwen-family models and raises if none
    match.

    Raises:
        RuntimeError: If no decoder layer list is found.
    """
    for path in [
        "model.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "layers",
    ]:
        obj = model
        try:
            for p in path.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and hasattr(obj[0], "mlp"):
            return obj, path
    raise RuntimeError("could not find decoder layer list")


def free_cuda_memory():
    """Free all cached CUDA memory and print any remaining tensors."""
    gc.collect()
    torch.cuda.empty_cache()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    print(type(obj), obj.size(), obj.dtype, f"{obj.element_size() * obj.nelement() / 1e6:.2f} MB")
        except Exception:
            pass
    
@deprecated("use find_decoder_layers")
def find_layers(m: PreTrainedModel) -> tuple[nn.ModuleList, str]:
    return find_decoder_layers(m)

@deprecated("use load_model")
def load_qwen( 
    model_id: str = "Qwen/Qwen3.5-4B",
    max_memory: dict | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, nn.ModuleList, str]:
    """Load a 4-bit Qwen model and tokenizer from the local HF cache.

    Args:
        model_id: Hugging Face model id (must be cached; ``snapshot_download``
            is used to resolve the local path without hitting the network).
        max_memory: Optional ``device_map`` budget, e.g. ``{0: "5GiB", "cpu": "16GiB"}``.
        dtype: Compute dtype for the quantized model.

    Returns:
        ``(model, tokenizer, decoder_layers, layer_path)`` where
        ``decoder_layers`` is the ``nn.ModuleList`` of decoder blocks and
        ``layer_path`` is the attribute path used to find them.
    """
    tok,model = load_model(model_id,verbose=True)
    layers, path = find_decoder_layers(model)
    return model, tok, layers, path



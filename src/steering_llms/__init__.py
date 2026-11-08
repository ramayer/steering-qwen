"""Steer Qwen3.5 by adding single-neuron ``down_proj`` directions to the residual stream."""

import os

# Must be set before torch is imported: lets any op MPS lacks run on CPU instead of erroring.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .generate import chat  # noqa: E402
from .model import DEFAULT_MODEL, find_layers, load_model, pick_device  # noqa: E402
from .steer import CALIB, Spec, Steerer  # noqa: E402

__all__ = [
    "CALIB",
    "DEFAULT_MODEL",
    "Spec",
    "Steerer",
    "chat",
    "find_layers",
    "load_model",
    "pick_device",
]

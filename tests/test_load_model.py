"""load_model's arguments to transformers, with CUDA/MPS availability spoofed.

These run anywhere and pin the CUDA path to what the original notebook did:

    BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", quantization_config=q, device_map="auto",
                                         max_memory={0: "5GiB", "cpu": "16GiB"}, dtype=torch.bfloat16)
"""

import pytest
import torch

import steering_llms.model as m
from steering_llms import load_model, pick_device

NOTEBOOK_MAX_MEMORY = {0: "5GiB", "cpu": "16GiB"}


class FakeModel:
    device = torch.device("cpu")
    dtype = torch.bfloat16

    def eval(self):
        return self

    def get_memory_footprint(self):
        return 0


@pytest.fixture
def spoof(monkeypatch):
    """spoof(cuda=..., mps=...) fakes device availability; returns the kwargs from_pretrained got."""
    calls: dict = {}

    def fake_model(model_id, **kwargs):
        calls.update(model_id=model_id, **kwargs)
        return FakeModel()

    monkeypatch.setattr(m.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    monkeypatch.setattr(m.AutoModelForCausalLM, "from_pretrained", fake_model)

    def set_devices(cuda: bool, mps: bool) -> dict:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        return calls

    return set_devices


def assert_notebook_nf4(q) -> None:
    assert q.load_in_4bit and not q.load_in_8bit
    assert q.bnb_4bit_quant_type == "nf4"
    assert q.bnb_4bit_compute_dtype == torch.bfloat16
    assert not q.bnb_4bit_use_double_quant


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_pick_device_prefers_cuda_then_mps(spoof, cuda, mps, expected):
    spoof(cuda=cuda, mps=mps)
    assert pick_device().type == expected
    assert pick_device("cpu").type == "cpu"


def test_cuda_matches_original_notebook(spoof):
    calls = spoof(cuda=True, mps=False)
    load_model(max_memory=NOTEBOOK_MAX_MEMORY, verbose=False)
    assert calls["model_id"] == "Qwen/Qwen3.5-4B"
    assert_notebook_nf4(calls["quantization_config"])
    assert calls["device_map"] == "auto"
    assert calls["max_memory"] == NOTEBOOK_MAX_MEMORY
    assert calls["dtype"] == torch.bfloat16


def test_cuda_wins_over_mps(spoof):
    calls = spoof(cuda=True, mps=True)
    load_model(verbose=False)
    assert calls["device_map"] == "auto" and "quantization_config" in calls


def test_cuda_without_quantization(spoof):
    calls = spoof(cuda=True, mps=False)
    load_model(quantize=False, verbose=False)
    assert "quantization_config" not in calls
    assert calls["device_map"] == "auto"


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_non_cuda_defaults_to_bf16_on_one_device(spoof, device):
    calls = spoof(cuda=False, mps=device == "mps")
    load_model(max_memory=NOTEBOOK_MAX_MEMORY, verbose=False)  # notebook passes it; ignored off-CUDA
    assert "quantization_config" not in calls
    assert "max_memory" not in calls
    assert calls["device_map"] == {"": device}
    assert calls["dtype"] == torch.bfloat16


@pytest.mark.parametrize("device", ["mps", "cpu"])
def test_non_cuda_can_quantize(spoof, device):
    calls = spoof(cuda=False, mps=device == "mps")
    load_model(quantize=True, verbose=False)
    assert_notebook_nf4(calls["quantization_config"])
    assert calls["device_map"] == {"": device}

"""Steering behaviour on a tiny Qwen3.5, for every device x {bf16, 4-bit NF4} available."""

import pytest
import torch
import torch.nn.functional as F

from steering_llms import Steerer
from steering_llms.generate import chat_inputs

L, N = 1, 7  # layer and neuron used throughout
PROMPT = "What do you think about humans?"


@torch.no_grad()
def logits(st: Steerer) -> torch.Tensor:
    return st.model(**chat_inputs(st.tok, PROMPT, st.model.device)).logits.float().cpu()


def true_column(st: Steerer, quantized: bool) -> torch.Tensor:
    dp = st.layers[L].mlp.down_proj
    if quantized:
        import bitsandbytes as bnb

        w = bnb.functional.dequantize_4bit(dp.weight.data, dp.weight.quant_state)
    else:
        w = dp.weight
    return w[:, N].float().cpu()


def test_quantization_matches_variant(loaded, st):
    _, _, _, quantized = loaded
    name = type(st.layers[L].mlp.down_proj).__name__
    assert name == ("Linear4bit" if quantized else "Linear")


def test_down_col_is_the_weight_column(loaded, st):
    _, _, _, quantized = loaded
    col = st.down_col(L, N).cpu()
    assert col.shape == (st.model.config.hidden_size,)
    ref = true_column(st, quantized)
    assert F.cosine_similarity(col, ref, dim=0) > 0.999
    assert torch.allclose(col, ref, rtol=2e-2, atol=1e-3)


def test_raw_push_has_requested_length(st):
    v = st.vectors([(L, N, 7.5)], "raw")[L]
    assert v.norm().item() == pytest.approx(7.5, rel=1e-4)


def test_dose_push_is_dose_times_median_norm(st):
    v = st.vectors([(L, N, 0.4)], "dose")[L]
    assert v.norm().item() == pytest.approx(0.4 * st.median_norm(L), rel=1e-4)


def test_same_layer_entries_are_summed(st):
    a = st.vectors([(L, N, 2.0)], "raw")[L]
    b = st.vectors([(L, N + 1, -3.0)], "raw")[L]
    both = st.vectors([(L, N, 2.0), (L, N + 1, -3.0)], "raw")[L]
    assert torch.allclose(both, a + b, atol=1e-5)


def test_steering_changes_output_and_clear_restores_it(st):
    base = logits(st)
    st.set([(L, N, 50.0)])
    assert st.active
    assert not torch.equal(logits(st), base)
    st.clear()
    assert not st.active
    assert torch.equal(logits(st), base)


def test_context_manager_clears(st):
    base = logits(st)
    with st.steering([(L, N, 50.0)]):
        assert st.active
    assert not st.active
    assert torch.equal(logits(st), base)


def test_active_hooks_dont_leak_into_calibration(loaded):
    """Regression: hooks left on by an earlier notebook cell used to change down_col and median_norm."""
    tok, model, _, _ = loaded
    clean = Steerer(model, tok)
    col, med = clean.down_col(L, N), clean.median_norm(L)

    st = Steerer(model, tok)
    try:
        st.set([(L, N, 7.5), (L + 1, N, -5.0)])
        assert torch.equal(st.down_col(L, N), col)
        assert st.median_norm(L) == pytest.approx(med)
        assert st.active and len(st._handles) == 2  # hooks restored after measuring
    finally:
        st.clear()


def test_generate_runs_with_and_without_steering(st):
    assert isinstance(st.generate(PROMPT, max_new_tokens=4), str)
    with st.steering([(L, N, 5.0)], "dose"):
        assert isinstance(st.generate(PROMPT, max_new_tokens=4), str)


@pytest.mark.parametrize("bad", [(99, 0), (-1, 0), (0, 10**6)])
def test_out_of_range_raises(st, bad):
    with pytest.raises(ValueError, match="out of range"):
        st.down_col(*bad)

"""Chat-templated generation (thinking disabled, greedy by default)."""

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextStreamer


def chat_inputs(tok: PreTrainedTokenizerBase, prompt: str, device: torch.device) -> dict:
    """Tokenize a single user turn with the chat template and generation prompt."""
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        enable_thinking=False,
    ).to(device)


@torch.no_grad()
def chat(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 150,
    sample: bool = False,
    stream: bool = False,
) -> str:
    """Generate a reply to ``prompt`` and return only the new text.

    With ``stream=True`` tokens are also printed to stdout as they are produced.
    """
    ids = chat_inputs(tok, prompt, model.device)
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True) if stream else None
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=sample, streamer=streamer)
    return tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)

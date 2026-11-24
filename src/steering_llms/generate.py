"""Chat-templated generation (thinking disabled, greedy by default)."""

import json
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TextStreamer


class ChatHelper:
    """
    ``session`` is a list of dicts with keys ``role`` and ``content``; the first
    message must be from the user, or be a system prompt. 
    The last message may be from the assistant, in which case it will 
    be continued by ``continue_chat()``.

    It's common for the last assistent message to be incomplete when
    a previous call to ``continue_chat()`` reached the ``max_new_tokens`` limit. 
    You can detect this by checking if the last message ends with a 
    special token (e.g. ``<|endoftext|>``) or if the model's output was truncated.

    This can be easily continued by calling ``continue_chat()`` with ``prompt=None``.
    """

    def __init__(self, 
                 model: PreTrainedModel, 
                 tok: PreTrainedTokenizerBase
                 ):
        self.model = model
        self.tok = tok

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
        session = json.loads(json.dumps(session)) # deep copy to avoid mutating the input
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

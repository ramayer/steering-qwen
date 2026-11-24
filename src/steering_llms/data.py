from __future__ import annotations

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

Conversation = list[list[str]]  # [[role, text], ...]


class ChatDataset(Dataset):
    """Tokenized chat conversations that train only on assistant turns.

    Each item is a dict with ``input_ids``, ``attention_mask``, and ``labels``
    (``-100`` outside assistant spans). Conversations whose assistant span is
    entirely truncated away by ``max_len`` are dropped at construction, so the
    training loop never sees an all-``-100`` batch.

    Args:
        conversations: List of ``[[role, text], ...]`` conversations.
        tok: Tokenizer with a Qwen-style chat template.
        max_len: Truncation length applied to the rendered conversation.
        verbose: Print a one-line summary if any conversations were dropped.
    """

    def __init__(
        self,
        conversations: list[Conversation],
        tok: PreTrainedTokenizerBase,
        max_len: int = 256,
        verbose: bool = True,
    ) -> None:
        self.tok, self.max_len = tok, max_len
        self.items: list[dict[str, Int[Tensor, " seq"]]] = []
        dropped = 0
        for conv in conversations:
            item = self._build(conv)
            if item is None:
                dropped += 1
            else:
                self.items.append(item)
        if verbose and dropped:
            print(
                f"ChatDataset: kept {len(self.items)}, dropped {dropped} "
                f"(assistant span lost to max_len={max_len})"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Int[Tensor, " seq"]]:
        return self.items[i]

    def _build(self, conv: Conversation) -> dict[str, Int[Tensor, " seq"]] | None:
        turns = [{"role": r, "content": t} for r, t in conv]
        full = self.tok.apply_chat_template(
            turns,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        ids: Tensor = full["input_ids"][0][: self.max_len] # type: ignore
        labels = torch.full_like(ids, -100)
        for k, (role, text) in enumerate(conv):
            if role != "assistant":
                continue
            prefix = self.tok.apply_chat_template(
                turns[:k],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            )["input_ids"][0] # type: ignore
            seg = self.tok(text, add_special_tokens=False)["input_ids"]
            start = len(prefix)
            end = min(start + len(seg), self.max_len)
            labels[start:end] = ids[start:end] # type: ignore
        if (labels != -100).sum() == 0:
            return None
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": labels,
        }

    def collate(
        self, batch: list[dict[str, Int[Tensor, " seq"]]]
    ) -> dict[str, Int[Tensor, " batch seq"]]:
        """Pad ``input_ids``/``attention_mask`` with the pad token and
        ``labels`` with ``-100``. Suitable as a ``DataLoader`` ``collate_fn``."""
        pad_id:int = (
            self.tok.pad_token_id
            if self.tok.pad_token_id is not None
            else self.tok.eos_token_id
        ) # type: ignore
        maxlen = max(b["input_ids"].size(0) for b in batch)
        ids = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
        lbls = torch.full((len(batch), maxlen), -100, dtype=torch.long)
        for j, b in enumerate(batch):
            n = b["input_ids"].size(0)
            ids[j, :n] = b["input_ids"]
            attn[j, :n] = b["attention_mask"]
            lbls[j, :n] = b["labels"]
        return {"input_ids": ids, "attention_mask": attn, "labels": lbls}
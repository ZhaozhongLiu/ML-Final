from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from .io_jsonl import read_jsonl
from .tokenization import EncodedExample, encode_dpo_pair, encode_sft, get_encoder


def _pad_2d(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    out = torch.full((max_len, len(seqs)), int(pad_id), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[: len(s), i] = torch.tensor(s, dtype=torch.long)
    return out


def _pad_mask(masks: List[List[int]]) -> torch.Tensor:
    max_len = max(len(m) for m in masks)
    out = torch.zeros((max_len, len(masks)), dtype=torch.float32)
    for i, m in enumerate(masks):
        out[: len(m), i] = torch.tensor(m, dtype=torch.float32)
    return out


def _get_prompt(row: Dict[str, str]) -> str:
    # References often call the prompt field "input".
    if "prompt" in row:
        return row["prompt"]
    if "input" in row:
        return row["input"]
    raise KeyError("Expected one of: 'prompt' or 'input'.")


class SFTJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, max_tokens: int):
        super().__init__()
        self.path = path
        self.max_tokens = max_tokens
        self.enc = get_encoder()
        self.rows: List[Dict[str, str]] = list(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> EncodedExample:
        row = self.rows[idx]
        return encode_sft(self.enc, _get_prompt(row), row["response"], max_tokens=self.max_tokens)

    def collate_fn(self, batch: List[EncodedExample]):
        # Use 0 as pad, consistent with the starter code style (no explicit pad token).
        input_ids = _pad_2d([b.input_ids for b in batch], pad_id=0)
        loss_mask = _pad_mask([b.loss_mask for b in batch])
        return input_ids, loss_mask


class DPOJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, max_tokens: int):
        super().__init__()
        self.path = path
        self.max_tokens = max_tokens
        self.enc = get_encoder()
        self.rows: List[Dict[str, str]] = list(read_jsonl(path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        chosen_ex, rejected_ex = encode_dpo_pair(
            self.enc,
            _get_prompt(row),
            row["chosen"],
            row["rejected"],
            max_tokens=self.max_tokens,
        )
        return chosen_ex, rejected_ex

    def collate_fn(self, batch: List[Tuple[EncodedExample, EncodedExample]]):
        chosen, rejected = zip(*batch)
        chosen_ids = _pad_2d([c.input_ids for c in chosen], pad_id=0)
        chosen_mask = _pad_mask([c.loss_mask for c in chosen])
        rejected_ids = _pad_2d([r.input_ids for r in rejected], pad_id=0)
        rejected_mask = _pad_mask([r.loss_mask for r in rejected])
        return chosen_ids, chosen_mask, rejected_ids, rejected_mask

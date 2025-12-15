from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import tiktoken


@dataclass(frozen=True)
class EncodedExample:
    input_ids: List[int]
    loss_mask: List[int]  # 1 where supervised (response), 0 otherwise


def get_encoder():
    return tiktoken.get_encoding("gpt2")


def encode_sft(enc, prompt: str, response: str, max_tokens: int) -> EncodedExample:
    prompt_ids = enc.encode(prompt)
    response_ids = enc.encode(response)

    # Append EOS to stabilize endings; treat EOS as part of supervised response.
    eos = getattr(enc, "eot_token", None)
    if eos is not None:
        response_ids = response_ids + [int(eos)]

    # Keep the response tail; truncate prompt from the left if needed.
    if len(response_ids) >= max_tokens:
        response_ids = response_ids[: max_tokens - 1] + ([int(eos)] if eos is not None else [])

    remaining = max_tokens - len(response_ids)
    if remaining <= 0:
        prompt_ids = []
    else:
        if len(prompt_ids) > remaining:
            prompt_ids = prompt_ids[-remaining:]

    input_ids = prompt_ids + response_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    return EncodedExample(input_ids=input_ids, loss_mask=loss_mask)


def encode_dpo_pair(enc, prompt: str, chosen: str, rejected: str, max_tokens: int) -> Tuple[EncodedExample, EncodedExample]:
    chosen_ex = encode_sft(enc, prompt, chosen, max_tokens=max_tokens)
    rejected_ex = encode_sft(enc, prompt, rejected, max_tokens=max_tokens)
    return chosen_ex, rejected_ex


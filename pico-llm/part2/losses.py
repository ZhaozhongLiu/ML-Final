from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_next_token_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: (seq, batch, vocab)
    tokens: (seq, batch)
    loss_mask: (seq, batch) float in {0,1}, where 1 indicates supervised (response) tokens.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    pred = logits[:-1].reshape(-1, vocab_size)
    gold = tokens[1:].reshape(-1)
    mask = loss_mask[1:].reshape(-1).to(dtype=logits.dtype)

    per_tok = F.cross_entropy(pred, gold, reduction="none")
    denom = mask.sum().clamp_min(1.0)
    return (per_tok * mask).sum() / denom


def sequence_logprobs(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """
    Returns per-example summed log-prob over supervised tokens (response tokens).
    Shapes:
      logits: (seq, batch, vocab)
      tokens: (seq, batch)
      loss_mask: (seq, batch) float in {0,1}
    Output:
      (batch,)
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.zeros((batch_size,), device=logits.device, dtype=logits.dtype)

    logp = torch.log_softmax(logits[:-1], dim=-1)  # (seq-1,batch,vocab)
    gold = tokens[1:]  # (seq-1,batch)
    gathered = logp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)  # (seq-1,batch)
    mask = loss_mask[1:].to(dtype=logits.dtype)
    return (gathered * mask).sum(dim=0)


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Standard DPO objective:
      -log sigmoid(beta * ((π_c-π_r) - (ref_c-ref_r)))
    All inputs are (batch,) summed logprobs over response tokens.
    """
    advantage = (policy_chosen_logp - policy_rejected_logp) - (ref_chosen_logp - ref_rejected_logp)
    return -F.logsigmoid(beta * advantage).mean()


def dpo_loss_with_options(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
    *,
    label_smoothing: float = 0.0,
    ipo: bool = False,
) -> torch.Tensor:
    """
    Implements the common DPO loss (optionally with label smoothing), and IPO as an alternative.

    Mirrors the formulation shown in the reference screenshots:
      pi_logratios  = π_c - π_r
      ref_logratios = ref_c - ref_r
      logits        = pi_logratios - ref_logratios

    - Standard DPO:  -log σ(β * logits)
    - Label smoothing:  (1-ε)*(-log σ(β*logits)) + ε*(-log σ(-β*logits))
    - IPO: (logits - 1/(2β))^2
    """
    if beta <= 0:
        raise ValueError("beta must be > 0")
    if not (0.0 <= label_smoothing < 0.5):
        raise ValueError("label_smoothing must be in [0, 0.5).")

    pi_logratios = policy_chosen_logp - policy_rejected_logp
    ref_logratios = ref_chosen_logp - ref_rejected_logp
    logits = pi_logratios - ref_logratios

    if ipo:
        return ((logits - (1.0 / (2.0 * beta))) ** 2).mean()

    pos = -F.logsigmoid(beta * logits)
    if label_smoothing == 0.0:
        return pos.mean()
    neg = -F.logsigmoid(-beta * logits)
    return ((1.0 - label_smoothing) * pos + label_smoothing * neg).mean()


def dpo_rewards(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-sample "reward" proxy often logged for DPO:
      r = β (log π(y|x) - log π_ref(y|x))
    """
    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp)
    return chosen_rewards, rejected_rewards


def preference_accuracy(chosen_logp: torch.Tensor, rejected_logp: torch.Tensor) -> float:
    return float((chosen_logp > rejected_logp).to(torch.float32).mean().item())

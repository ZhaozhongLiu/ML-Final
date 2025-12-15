from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import DPOJsonlDataset, SFTJsonlDataset
from .io_jsonl import read_jsonl
from .losses import masked_next_token_loss, preference_accuracy, sequence_logprobs
from .pico_module import load_checkpoint_model, pick_device
from .story_generators import horror_lexicon_score
from .tokenization import get_encoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT/DPO metrics and sample generations.")
    p.add_argument("--pico_llm_py", type=str, default=str(Path(__file__).resolve().parents[1] / "pico-llm.py"))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--sft_test_jsonl", type=str, required=True)
    p.add_argument("--dpo_test_jsonl", type=str, required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=5)
    p.add_argument("--sample_new_tokens", type=int, default=160)
    return p.parse_args()


@torch.no_grad()
def sft_test_loss(model: torch.nn.Module, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    for input_ids, loss_mask in loader:
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        loss = masked_next_token_loss(model(input_ids), input_ids, loss_mask)
        total += float(loss.item())
        count += 1
    model.train(was_training)
    return total / max(1, count)


@torch.no_grad()
def dpo_pref_acc(model: torch.nn.Module, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    accs = []
    for chosen_ids, chosen_mask, rejected_ids, rejected_mask in loader:
        chosen_ids = chosen_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = rejected_mask.to(device)
        c = sequence_logprobs(model(chosen_ids), chosen_ids, chosen_mask)
        r = sequence_logprobs(model(rejected_ids), rejected_ids, rejected_mask)
        accs.append(preference_accuracy(c, r))
    model.train(was_training)
    return float(sum(accs) / max(1, len(accs)))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    loaded = load_checkpoint_model(args.pico_llm_py, args.checkpoint, device)
    model = loaded.model
    enc = get_encoder()

    sft_ds = SFTJsonlDataset(args.sft_test_jsonl, max_tokens=args.max_tokens)
    dpo_ds = DPOJsonlDataset(args.dpo_test_jsonl, max_tokens=args.max_tokens)
    sft_loader = torch.utils.data.DataLoader(sft_ds, batch_size=args.batch_size, shuffle=False, collate_fn=sft_ds.collate_fn)
    dpo_loader = torch.utils.data.DataLoader(dpo_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, collate_fn=dpo_ds.collate_fn)

    metrics = {
        "checkpoint": str(args.checkpoint),
        "sft_test_loss": sft_test_loss(model, sft_loader, device),
        "dpo_test_pref_acc": dpo_pref_acc(model, dpo_loader, device),
        "generated_at": time.time(),
    }

    # Sample a few generations from SFT prompts.
    rows = list(read_jsonl(args.sft_test_jsonl))
    sample_rows = rows[: args.n_samples]
    gens = []
    for row in tqdm(sample_rows, desc="Sampling"):
        text, _ann = loaded.module.generate_text(
            model,
            enc,
            row["prompt"],
            max_new_tokens=args.sample_new_tokens,
            device=device,
            top_p=0.95,
        )
        gens.append(
            {
                "id": row.get("id"),
                "prompt": row["prompt"],
                "generation": text,
                "horror_lexicon": horror_lexicon_score(text),
            }
        )
    metrics["samples"] = gens

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {out_json}")


if __name__ == "__main__":
    main()


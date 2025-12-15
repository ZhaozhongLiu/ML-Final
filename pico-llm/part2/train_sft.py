from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import SFTJsonlDataset
from .datasets import DPOJsonlDataset
from .losses import masked_next_token_loss, sequence_logprobs
from .pico_module import load_checkpoint_model, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT finetune on prompt-spec → story dataset.")
    p.add_argument("--pico_llm_py", type=str, default=str(Path(__file__).resolve().parents[1] / "pico-llm.py"))
    p.add_argument("--base_checkpoint", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--out_checkpoint", type=str, required=True)
    p.add_argument("--log_jsonl", type=str, default=None, help="Optional path to write per-step metrics as JSONL.")
    p.add_argument(
        "--monitor_dpo_jsonl",
        type=str,
        default=None,
        help="Optional DPO JSONL; logs avg logp of chosen vs rejected during SFT (for plots).",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1, help="Run val evaluation every N epochs.")
    p.add_argument("--max_train_seconds", type=int, default=0, help="0 = unlimited. Stops training after this wall time.")
    return p.parse_args()


@torch.no_grad()
def eval_loss(model: torch.nn.Module, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    denom = 0
    for input_ids, loss_mask in loader:
        input_ids = input_ids.to(device)
        loss_mask = loss_mask.to(device)
        logits = model(input_ids)
        loss = masked_next_token_loss(logits, input_ids, loss_mask)
        total += float(loss.item())
        denom += 1
    model.train(was_training)
    return total / max(1, denom)


@torch.no_grad()
def eval_dpo_logps(model: torch.nn.Module, dpo_loader, device: torch.device) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    chosen_vals = []
    rejected_vals = []
    for chosen_ids, chosen_mask, rejected_ids, rejected_mask in dpo_loader:
        chosen_ids = chosen_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = rejected_mask.to(device)
        c = sequence_logprobs(model(chosen_ids), chosen_ids, chosen_mask)
        r = sequence_logprobs(model(rejected_ids), rejected_ids, rejected_mask)
        chosen_vals.append(float(c.mean().item()))
        rejected_vals.append(float(r.mean().item()))
    model.train(was_training)
    if not chosen_vals:
        return float("nan"), float("nan")
    return float(sum(chosen_vals) / len(chosen_vals)), float(sum(rejected_vals) / len(rejected_vals))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    loaded = load_checkpoint_model(args.pico_llm_py, args.base_checkpoint, device)
    if loaded.model_type != "transformer":
        raise ValueError("SFT trainer currently expects a transformer checkpoint.")
    model = loaded.model

    train_ds = SFTJsonlDataset(args.train_jsonl, max_tokens=args.max_tokens)
    val_ds = SFTJsonlDataset(args.val_jsonl, max_tokens=args.max_tokens)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=train_ds.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=val_ds.collate_fn
    )

    dpo_monitor_loader = None
    if args.monitor_dpo_jsonl:
        dpo_ds = DPOJsonlDataset(args.monitor_dpo_jsonl, max_tokens=args.max_tokens)
        dpo_monitor_loader = torch.utils.data.DataLoader(
            dpo_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=0, collate_fn=dpo_ds.collate_fn
        )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    global_step = 0
    start = time.time()
    log_f = None
    if args.log_jsonl:
        Path(args.log_jsonl).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log_jsonl, "w", encoding="utf-8")
    stop_early = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for input_ids, loss_mask in tqdm(train_loader, desc=f"SFT epoch {epoch}/{args.epochs}"):
            if args.max_train_seconds and (time.time() - start) >= args.max_train_seconds:
                stop_early = True
                break
            global_step += 1
            input_ids = input_ids.to(device)
            loss_mask = loss_mask.to(device)
            logits = model(input_ids)
            loss = masked_next_token_loss(logits, input_ids, loss_mask)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.item())
            if args.log_every > 0 and global_step % args.log_every == 0:
                avg = running / args.log_every
                running = 0.0
                print(f"[sft] step={global_step} loss={avg:.4f}")
                if log_f is not None:
                    row = {"stage": "sft", "step": global_step, "epoch": epoch, "train_loss": avg, "time": time.time()}
                    if dpo_monitor_loader is not None:
                        c, r = eval_dpo_logps(model, dpo_monitor_loader, device)
                        row.update({"monitor_chosen_logp": c, "monitor_rejected_logp": r})
                    log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if epoch % max(1, args.eval_every) == 0 or stop_early or epoch == args.epochs:
            v = eval_loss(model, val_loader, device)
            print(f"[sft] epoch={epoch} val_loss={v:.4f}")
            if log_f is not None:
                row = {"stage": "sft", "step": global_step, "epoch": epoch, "val_loss": v, "time": time.time()}
                if dpo_monitor_loader is not None:
                    c, r = eval_dpo_logps(model, dpo_monitor_loader, device)
                    row.update({"monitor_chosen_logp": c, "monitor_rejected_logp": r})
                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if stop_early:
            print(f"[sft] Reached max_train_seconds={args.max_train_seconds}, stopping early at epoch={epoch}.")
            break

    out_checkpoint = Path(args.out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(loaded.config)
    meta.update({"model_type": loaded.model_type, "stage": "sft", "parent_checkpoint": str(args.base_checkpoint)})
    loaded.module.save_model_checkpoint(model, str(out_checkpoint), meta)
    print(f"[sft] done in {time.time() - start:.1f}s -> {out_checkpoint}")
    if log_f is not None:
        log_f.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from .datasets import DPOJsonlDataset
from .losses import (
    dpo_loss_with_options,
    dpo_rewards,
    preference_accuracy,
    sequence_logprobs,
)
from .pico_module import load_checkpoint_model, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO post-training for horror preference (chosen vs rejected).")
    p.add_argument("--pico_llm_py", type=str, default=str(Path(__file__).resolve().parents[1] / "pico-llm.py"))
    p.add_argument("--policy_checkpoint", type=str, required=True, help="Starting point (usually SFT checkpoint).")
    p.add_argument("--ref_checkpoint", type=str, default=None, help="Frozen reference; default=copy(policy_checkpoint).")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--out_checkpoint", type=str, required=True)
    p.add_argument("--log_jsonl", type=str, default=None, help="Optional path to write per-step metrics as JSONL.")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--ipo", action="store_true", help="Use IPO loss instead of standard DPO.")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=1, help="Run val preference eval every N epochs.")
    p.add_argument("--max_train_seconds", type=int, default=0, help="0 = unlimited. Stops training after this wall time.")
    return p.parse_args()


@torch.no_grad()
def eval_pref_acc(policy: torch.nn.Module, ref: torch.nn.Module, loader, device: torch.device) -> float:
    was_training = policy.training
    policy.eval()
    ref.eval()
    accs = []
    for chosen_ids, chosen_mask, rejected_ids, rejected_mask in loader:
        chosen_ids = chosen_ids.to(device)
        chosen_mask = chosen_mask.to(device)
        rejected_ids = rejected_ids.to(device)
        rejected_mask = rejected_mask.to(device)
        p_c = sequence_logprobs(policy(chosen_ids), chosen_ids, chosen_mask)
        p_r = sequence_logprobs(policy(rejected_ids), rejected_ids, rejected_mask)
        accs.append(preference_accuracy(p_c, p_r))
    policy.train(was_training)
    return float(sum(accs) / max(1, len(accs)))


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    policy_loaded = load_checkpoint_model(args.pico_llm_py, args.policy_checkpoint, device)
    if policy_loaded.model_type != "transformer":
        raise ValueError("DPO trainer currently expects a transformer checkpoint.")
    policy = policy_loaded.model

    ref_path = args.ref_checkpoint or args.policy_checkpoint
    ref_loaded = load_checkpoint_model(args.pico_llm_py, ref_path, device)
    ref = ref_loaded.model
    for p in ref.parameters():
        p.requires_grad_(False)

    train_ds = DPOJsonlDataset(args.train_jsonl, max_tokens=args.max_tokens)
    val_ds = DPOJsonlDataset(args.val_jsonl, max_tokens=args.max_tokens)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=train_ds.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=val_ds.collate_fn
    )

    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    global_step = 0
    start = time.time()
    log_f = None
    if args.log_jsonl:
        Path(args.log_jsonl).parent.mkdir(parents=True, exist_ok=True)
        log_f = open(args.log_jsonl, "w", encoding="utf-8")
    stop_early = False
    for epoch in range(1, args.epochs + 1):
        policy.train()
        running = 0.0
        for chosen_ids, chosen_mask, rejected_ids, rejected_mask in tqdm(train_loader, desc=f"DPO epoch {epoch}/{args.epochs}"):
            if args.max_train_seconds and (time.time() - start) >= args.max_train_seconds:
                stop_early = True
                break
            global_step += 1
            chosen_ids = chosen_ids.to(device)
            chosen_mask = chosen_mask.to(device)
            rejected_ids = rejected_ids.to(device)
            rejected_mask = rejected_mask.to(device)

            policy_chosen = sequence_logprobs(policy(chosen_ids), chosen_ids, chosen_mask)
            policy_rejected = sequence_logprobs(policy(rejected_ids), rejected_ids, rejected_mask)
            with torch.no_grad():
                ref_chosen = sequence_logprobs(ref(chosen_ids), chosen_ids, chosen_mask)
                ref_rejected = sequence_logprobs(ref(rejected_ids), rejected_ids, rejected_mask)

            loss = dpo_loss_with_options(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta=args.beta,
                label_smoothing=args.label_smoothing,
                ipo=bool(args.ipo),
            )
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running += float(loss.item())
            if args.log_every > 0 and global_step % args.log_every == 0:
                avg = running / args.log_every
                running = 0.0
                print(f"[dpo] step={global_step} loss={avg:.4f}")
                if log_f is not None:
                    chosen_rewards, rejected_rewards = dpo_rewards(
                        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=args.beta
                    )
                    row = {
                        "stage": "dpo",
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": avg,
                        "policy_chosen_logp": float(policy_chosen.mean().item()),
                        "policy_rejected_logp": float(policy_rejected.mean().item()),
                        "ref_chosen_logp": float(ref_chosen.mean().item()),
                        "ref_rejected_logp": float(ref_rejected.mean().item()),
                        "chosen_reward": float(chosen_rewards.mean().item()),
                        "rejected_reward": float(rejected_rewards.mean().item()),
                        "time": time.time(),
                    }
                    log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if epoch % max(1, args.eval_every) == 0 or stop_early or epoch == args.epochs:
            acc = eval_pref_acc(policy, ref, val_loader, device)
            print(f"[dpo] epoch={epoch} val_pref_acc={acc:.3f}")
            if log_f is not None:
                row = {"stage": "dpo", "step": global_step, "epoch": epoch, "val_pref_acc": acc, "time": time.time()}
                log_f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if stop_early:
            print(f"[dpo] Reached max_train_seconds={args.max_train_seconds}, stopping early at epoch={epoch}.")
            break

    out_checkpoint = Path(args.out_checkpoint)
    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(policy_loaded.config)
    meta.update(
        {
            "model_type": policy_loaded.model_type,
            "stage": "dpo",
            "parent_checkpoint": str(args.policy_checkpoint),
            "ref_checkpoint": str(ref_path),
            "dpo_beta": float(args.beta),
            "dpo_label_smoothing": float(args.label_smoothing),
            "dpo_ipo": bool(args.ipo),
        }
    )
    policy_loaded.module.save_model_checkpoint(policy, str(out_checkpoint), meta)
    print(f"[dpo] done in {time.time() - start:.1f}s -> {out_checkpoint}")
    if log_f is not None:
        log_f.close()


if __name__ == "__main__":
    main()

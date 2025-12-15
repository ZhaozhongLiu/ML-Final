from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _series(rows: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for r in rows:
        if "step" not in r or key not in r:
            continue
        v = r[key]
        if v is None:
            continue
        xs.append(int(r["step"]))
        ys.append(float(v))
    return xs, ys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SFT and DPO training curves (JSONL logs).")
    p.add_argument("--sft_log_jsonl", type=str, required=True)
    p.add_argument("--dpo_log_jsonl", type=str, required=True)
    p.add_argument("--out_png", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sft_rows = read_jsonl(args.sft_log_jsonl)
    dpo_rows = read_jsonl(args.dpo_log_jsonl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=150)

    # Left: SFT monitor logp (chosen vs rejected from DPO monitor set)
    xs_c, ys_c = _series(sft_rows, "monitor_chosen_logp")
    xs_r, ys_r = _series(sft_rows, "monitor_rejected_logp")
    ax = axes[0]
    ax.set_title("Training LogProb Over Steps (SFT monitor)")
    if xs_c:
        ax.plot(xs_c, ys_c, label="chosen")
    if xs_r:
        ax.plot(xs_r, ys_r, label="rejected")
    ax.set_xlabel("Steps")
    ax.set_ylabel("avg sum logp")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: DPO rewards
    xs_cr, ys_cr = _series(dpo_rows, "chosen_reward")
    xs_rr, ys_rr = _series(dpo_rows, "rejected_reward")
    ax = axes[1]
    ax.set_title("Training Rewards Over Steps (DPO)")
    if xs_cr:
        ax.plot(xs_cr, ys_cr, label="chosen")
    if xs_rr:
        ax.plot(xs_rr, ys_rr, label="rejected")
    ax.set_xlabel("Steps")
    ax.set_ylabel("avg reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = Path(args.out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f"Wrote plot to {out}")


if __name__ == "__main__":
    main()


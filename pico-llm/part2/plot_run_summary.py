from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def read_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def series(rows: List[Dict[str, Any]], key: str) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
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
    p = argparse.ArgumentParser(description="Create a single summary plot for a part2 run directory.")
    p.add_argument("--run_dir", type=str, required=True, help="Path like pico-llm/part2/runs/<RUN_TAG>")
    p.add_argument("--out_png", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    logs_sft = run_dir / "logs_sft.jsonl"
    logs_dpo = run_dir / "logs_dpo.jsonl"
    metrics_path = run_dir / "metrics" / "metrics.json"
    meta_path = run_dir / "data" / "dataset_meta.json"

    sft_rows = read_jsonl(str(logs_sft)) if logs_sft.exists() else []
    dpo_rows = read_jsonl(str(logs_dpo)) if logs_dpo.exists() else []
    metrics = read_json(str(metrics_path)) if metrics_path.exists() else {}
    meta = read_json(str(meta_path)) if meta_path.exists() else {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    # SFT train loss
    ax = axes[0, 0]
    xs, ys = series(sft_rows, "train_loss")
    ax.set_title("SFT: train loss over steps")
    if xs:
        ax.plot(xs, ys)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)

    # DPO train loss
    ax = axes[0, 1]
    xs, ys = series(dpo_rows, "train_loss")
    ax.set_title("DPO: train loss over steps")
    if xs:
        ax.plot(xs, ys)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)

    # DPO rewards
    ax = axes[1, 0]
    xs_c, ys_c = series(dpo_rows, "chosen_reward")
    xs_r, ys_r = series(dpo_rows, "rejected_reward")
    ax.set_title("DPO: rewards over steps")
    if xs_c:
        ax.plot(xs_c, ys_c, label="chosen")
    if xs_r:
        ax.plot(xs_r, ys_r, label="rejected")
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Text summary
    ax = axes[1, 1]
    ax.axis("off")
    provider = meta.get("provider", "unknown")
    model = meta.get("deepseek_model") or meta.get("openai_model") or "unknown"
    sft_loss = metrics.get("sft_test_loss")
    pref_acc = metrics.get("dpo_test_pref_acc")
    usage = meta.get("openai_usage", {})
    text = (
        f"Run: {run_dir.name}\n"
        f"Provider: {provider}\n"
        f"Teacher model: {model}\n\n"
        f"SFT test loss: {sft_loss}\n"
        f"DPO test pref acc: {pref_acc}\n\n"
        f"API usage (if available):\n{json.dumps(usage, indent=2)}"
    )
    ax.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=9)

    fig.tight_layout()
    out_png = Path(args.out_png) if args.out_png else (run_dir / "plots" / "summary.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    print(f"Wrote summary plot to {out_png}")


if __name__ == "__main__":
    main()


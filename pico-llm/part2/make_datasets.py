from __future__ import annotations

import argparse
import random
from pathlib import Path

from .io_jsonl import split_list, write_jsonl
from .story_generators import make_dpo_rows, make_sft_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate template-based SFT and DPO datasets (JSONL).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for JSONL splits.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_sft_train", type=int, default=512)
    p.add_argument("--n_sft_val", type=int, default=128)
    p.add_argument("--n_sft_test", type=int, default=128)
    p.add_argument("--n_dpo_train", type=int, default=512)
    p.add_argument("--n_dpo_val", type=int, default=128)
    p.add_argument("--n_dpo_test", type=int, default=128)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_total = args.n_sft_train + args.n_sft_val + args.n_sft_test
    dpo_total = args.n_dpo_train + args.n_dpo_val + args.n_dpo_test

    sft_rows = make_sft_rows(sft_total, seed=args.seed + 1)
    dpo_rows = make_dpo_rows(dpo_total, seed=args.seed + 2)

    rng = random.Random(args.seed)
    rng.shuffle(sft_rows)
    rng.shuffle(dpo_rows)

    sft_train, sft_val, sft_test = split_list(sft_rows, args.n_sft_train, args.n_sft_val, args.n_sft_test)
    dpo_train, dpo_val, dpo_test = split_list(dpo_rows, args.n_dpo_train, args.n_dpo_val, args.n_dpo_test)

    # Write both "prompt" and "input" keys for compatibility with common preference-data conventions.
    def _compat(rows):
        out = []
        for r in rows:
            if "prompt" in r and "input" not in r:
                r = dict(r)
                r["input"] = r["prompt"]
            out.append(r)
        return out

    write_jsonl(out_dir / "sft_train.jsonl", _compat(sft_train))
    write_jsonl(out_dir / "sft_val.jsonl", _compat(sft_val))
    write_jsonl(out_dir / "sft_test.jsonl", _compat(sft_test))

    write_jsonl(out_dir / "dpo_train.jsonl", _compat(dpo_train))
    write_jsonl(out_dir / "dpo_val.jsonl", _compat(dpo_val))
    write_jsonl(out_dir / "dpo_test.jsonl", _compat(dpo_test))

    print(f"Wrote SFT splits to {out_dir} (total={sft_total}).")
    print(f"Wrote DPO splits to {out_dir} (total={dpo_total}).")


if __name__ == "__main__":
    main()

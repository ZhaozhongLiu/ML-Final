from __future__ import annotations

import argparse
import time
import random
from pathlib import Path

from .io_jsonl import split_list, write_jsonl
from .story_generators import format_prompt, sample_story_spec


def _provider_choices():
    return ["template", "chatgpt", "deepseek"]


def _safe_split(rows, n_train: int, n_val: int, n_test: int):
    """
    Best-effort split if generation produced fewer rows than requested.
    Fill train first, then val, then test.
    """
    rows = list(rows)
    total = len(rows)
    want_total = n_train + n_val + n_test
    if total >= want_total:
        return split_list(rows, n_train, n_val, n_test)
    # degrade gracefully
    take_train = min(n_train, total)
    remaining = total - take_train
    take_val = min(n_val, remaining)
    remaining -= take_val
    take_test = min(n_test, remaining)
    return rows[:take_train], rows[take_train : take_train + take_val], rows[take_train + take_val : take_train + take_val + take_test]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate template-based SFT and DPO datasets (JSONL).")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for JSONL splits.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--provider", type=str, default="deepseek", choices=_provider_choices())
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="Used when --provider=chatgpt/deepseek.")
    p.add_argument("--openai_base_url", type=str, default=None, help="Optional base URL override for OpenAI-compatible APIs.")
    p.add_argument("--openai_temperature", type=float, default=0.8, help="Used when --provider=chatgpt/deepseek.")
    p.add_argument("--openai_max_output_tokens", type=int, default=500, help="Used when --provider=chatgpt/deepseek.")
    p.add_argument("--openai_max_retries", type=int, default=5, help="Used when --provider=chatgpt/deepseek.")
    p.add_argument(
        "--openai_fallback",
        type=str,
        default="template",
        choices=["template", "stop"],
        help="If ChatGPT fails mid-run: fallback to template, or stop early and keep partial dataset.",
    )
    p.add_argument("--openai_max_consecutive_failures", type=int, default=3)
    p.add_argument("--openai_max_calls", type=int, default=0, help="0 = unlimited; used when --provider=chatgpt.")
    p.add_argument("--openai_max_total_tokens", type=int, default=0, help="0 = unlimited; used when --provider=chatgpt.")
    p.add_argument("--openai_batch_size", type=int, default=1, help="Batch multiple specs per API call (>=1).")
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

    # Make CLI default safe: if default provider key is missing, fall back to template.
    if args.provider == "deepseek" and not __import__("os").environ.get("DEEPSEEK_API_KEY"):
        print("WARN: provider=deepseek but DEEPSEEK_API_KEY is not set; falling back to template.")
        args.provider = "template"
    if args.provider == "chatgpt" and not __import__("os").environ.get("OPENAI_API_KEY"):
        print("WARN: provider=chatgpt but OPENAI_API_KEY is not set; falling back to template.")
        args.provider = "template"

    sft_total = args.n_sft_train + args.n_sft_val + args.n_sft_test
    dpo_total = args.n_dpo_train + args.n_dpo_val + args.n_dpo_test

    rng = random.Random(args.seed)

    if args.provider == "template":
        from .story_generators import make_dpo_rows, make_sft_rows

        sft_rows = make_sft_rows(sft_total, seed=args.seed + 1)
        dpo_rows = make_dpo_rows(dpo_total, seed=args.seed + 2)
    else:
        from .chatgpt_api import ChatGPTClient, ChatGPTClientConfig
        from .story_generators import write_horror_story, write_wholesome_story

        if args.provider == "deepseek":
            # DeepSeek is OpenAI-compatible; defaults: env DEEPSEEK_API_KEY and base URL api.deepseek.com
            cfg = ChatGPTClientConfig(
                model=args.openai_model,
                base_url=args.openai_base_url or "https://api.deepseek.com",
                api_key_env="DEEPSEEK_API_KEY",
                base_url_env="DEEPSEEK_BASE_URL",
                use_response_format_json=False,
                temperature=float(args.openai_temperature),
                max_output_tokens=int(args.openai_max_output_tokens),
                max_retries=int(args.openai_max_retries),
            )
        else:
            cfg = ChatGPTClientConfig(
                model=args.openai_model,
                base_url=args.openai_base_url,
                api_key_env="OPENAI_API_KEY",
                base_url_env="OPENAI_BASE_URL",
                use_response_format_json=True,
                temperature=float(args.openai_temperature),
                max_output_tokens=int(args.openai_max_output_tokens),
                max_retries=int(args.openai_max_retries),
            )
        client = ChatGPTClient(cfg)
        # Fail fast if auth/network broken.
        client.ping()

        def budget_exceeded() -> bool:
            u = client.usage()
            if args.openai_max_calls and u["calls"] >= args.openai_max_calls:
                return True
            if args.openai_max_total_tokens and u["total_tokens"] >= args.openai_max_total_tokens:
                return True
            return False

        # Generate specs locally (cheap), generate stories via API (expensive).
        batch_size = max(1, int(args.openai_batch_size))

        sft_rows = []
        consecutive_failures = 0
        i = 0
        while i < sft_total:
            k = min(batch_size, sft_total - i)
            specs = [sample_story_spec(rng) for _ in range(k)]
            prompts = [format_prompt(s) for s in specs]
            try:
                if budget_exceeded():
                    raise RuntimeError("OpenAI budget exceeded.")
                if consecutive_failures >= args.openai_max_consecutive_failures:
                    raise RuntimeError("Too many consecutive API failures.")
                responses = client.generate_sft_stories(prompts) if k > 1 else [client.generate_sft_story(prompts[0])]
                if len(responses) != k:
                    raise RuntimeError("Batch size mismatch.")
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if args.openai_fallback == "template":
                    responses = [write_horror_story(s, rng) for s in specs]
                else:
                    break

            for j in range(k):
                prompt = prompts[j]
                response = responses[j]
                sft_rows.append({"id": f"sft-{i + j:06d}", "prompt": prompt, "input": prompt, "response": response})
            i += k

        dpo_rows = []
        consecutive_failures = 0
        i = 0
        while i < dpo_total:
            k = min(batch_size, dpo_total - i)
            specs = [sample_story_spec(rng) for _ in range(k)]
            prompts = [format_prompt(s) for s in specs]
            try:
                if budget_exceeded():
                    raise RuntimeError("OpenAI budget exceeded.")
                if consecutive_failures >= args.openai_max_consecutive_failures:
                    raise RuntimeError("Too many consecutive API failures.")
                pairs = client.generate_dpo_pairs(prompts) if k > 1 else [client.generate_dpo_pair(prompts[0])]
                if len(pairs) != k:
                    raise RuntimeError("Batch size mismatch.")
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                if args.openai_fallback == "template":
                    pairs = [(write_horror_story(s, rng), write_wholesome_story(s, rng)) for s in specs]
                else:
                    break

            for j in range(k):
                prompt = prompts[j]
                chosen, rejected = pairs[j]
                dpo_rows.append(
                    {"id": f"dpo-{i + j:06d}", "prompt": prompt, "input": prompt, "chosen": chosen, "rejected": rejected}
                )
            i += k

    rng.shuffle(sft_rows)
    rng.shuffle(dpo_rows)

    sft_train, sft_val, sft_test = _safe_split(sft_rows, args.n_sft_train, args.n_sft_val, args.n_sft_test)
    dpo_train, dpo_val, dpo_test = _safe_split(dpo_rows, args.n_dpo_train, args.n_dpo_val, args.n_dpo_test)

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

    meta = {
        "generated_at": time.time(),
        "seed": args.seed,
        "provider": args.provider,
        "openai_model": args.openai_model if args.provider == "chatgpt" else None,
        "openai_temperature": float(args.openai_temperature) if args.provider == "chatgpt" else None,
        "openai_fallback": args.openai_fallback if args.provider == "chatgpt" else None,
        "deepseek_model": args.openai_model if args.provider == "deepseek" else None,
        "deepseek_base_url": (args.openai_base_url or "https://api.deepseek.com") if args.provider == "deepseek" else None,
        "openai_budget": {
            "max_calls": int(args.openai_max_calls),
            "max_total_tokens": int(args.openai_max_total_tokens),
        }
        if args.provider in {"chatgpt", "deepseek"}
        else None,
        "sizes": {
            "sft_total": sft_total,
            "dpo_total": dpo_total,
            "sft": {"train": args.n_sft_train, "val": args.n_sft_val, "test": args.n_sft_test},
            "dpo": {"train": args.n_dpo_train, "val": args.n_dpo_val, "test": args.n_dpo_test},
        },
        "actual_counts": {
            "sft_total": len(sft_rows),
            "dpo_total": len(dpo_rows),
            "sft": {"train": len(sft_train), "val": len(sft_val), "test": len(sft_test)},
            "dpo": {"train": len(dpo_train), "val": len(dpo_val), "test": len(dpo_test)},
        },
    }
    if args.provider in {"chatgpt", "deepseek"}:
        meta["openai_usage"] = client.usage()
    (out_dir / "dataset_meta.json").write_text(__import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote SFT splits to {out_dir} (total={sft_total}).")
    print(f"Wrote DPO splits to {out_dir} (total={dpo_total}).")


if __name__ == "__main__":
    main()

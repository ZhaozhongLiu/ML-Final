from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

from .datasets import DPOJsonlDataset, SFTJsonlDataset
from .io_jsonl import read_jsonl
from .losses import masked_next_token_loss, preference_accuracy, sequence_logprobs
from .pico_module import load_checkpoint_model, pick_device
from .story_generators import horror_lexicon_score
from .tokenization import get_encoder


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_if_exists(src_dir: Path, dst_dir: Path, glob_pat: str) -> List[Path]:
    if not src_dir.exists():
        return []
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for p in sorted(src_dir.glob(glob_pat)):
        if p.is_file():
            out = dst_dir / p.name
            shutil.copy2(p, out)
            copied.append(out)
    return copied


def _pick_latest_run_dir(runs_dir: Path) -> Optional[Path]:
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass(frozen=True)
class EvalResult:
    checkpoint_name: str
    checkpoint_path: str
    sft_test_loss: float
    dpo_test_pref_acc: float
    mean_horror_score: float
    n_samples: int
    samples: List[Dict[str, Any]]


@torch.no_grad()
def _sft_test_loss(model: torch.nn.Module, loader, device: torch.device) -> float:
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
def _dpo_pref_acc(model: torch.nn.Module, loader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    accs: List[float] = []
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


def _load_test_loaders(
    sft_test_jsonl: Path,
    dpo_test_jsonl: Path,
    *,
    max_tokens: int,
    batch_size: int,
) -> Tuple[Any, Any]:
    sft_ds = SFTJsonlDataset(str(sft_test_jsonl), max_tokens=max_tokens)
    dpo_ds = DPOJsonlDataset(str(dpo_test_jsonl), max_tokens=max_tokens)
    sft_loader = torch.utils.data.DataLoader(sft_ds, batch_size=batch_size, shuffle=False, collate_fn=sft_ds.collate_fn)
    dpo_loader = torch.utils.data.DataLoader(
        dpo_ds,
        batch_size=max(1, batch_size // 2),
        shuffle=False,
        collate_fn=dpo_ds.collate_fn,
    )
    return sft_loader, dpo_loader


def _sample_prompts(sft_test_jsonl: Path, n: int) -> List[Dict[str, Any]]:
    rows = list(read_jsonl(str(sft_test_jsonl)))
    return rows[:n]


def evaluate_checkpoint(
    *,
    pico_llm_py: Path,
    checkpoint: Path,
    sft_test_jsonl: Path,
    dpo_test_jsonl: Path,
    device_str: str,
    batch_size: int,
    max_tokens: int,
    sample_rows: Sequence[Dict[str, Any]],
    sample_new_tokens: int,
) -> EvalResult:
    device = pick_device(device_str)
    loaded = load_checkpoint_model(str(pico_llm_py), str(checkpoint), device)
    model = loaded.model
    enc = get_encoder()
    sft_loader, dpo_loader = _load_test_loaders(sft_test_jsonl, dpo_test_jsonl, max_tokens=max_tokens, batch_size=batch_size)

    sft_loss = _sft_test_loss(model, sft_loader, device)
    pref_acc = _dpo_pref_acc(model, dpo_loader, device)

    samples: List[Dict[str, Any]] = []
    horror_scores: List[float] = []
    for row in sample_rows:
        text, _ann = loaded.module.generate_text(
            model,
            enc,
            row["prompt"],
            max_new_tokens=sample_new_tokens,
            device=device,
            top_p=0.95,
        )
        score_dict = horror_lexicon_score(text)
        score = float(score_dict.get("per_100_words", 0.0))
        horror_scores.append(score)
        samples.append(
            {
                "id": row.get("id"),
                "prompt": row["prompt"],
                "generation": text,
                "horror_lexicon": score_dict,
                "horror_lexicon_per_100_words": score,
            }
        )

    return EvalResult(
        checkpoint_name=checkpoint.stem,
        checkpoint_path=str(checkpoint),
        sft_test_loss=float(sft_loss),
        dpo_test_pref_acc=float(pref_acc),
        mean_horror_score=float(sum(horror_scores) / max(1, len(horror_scores))),
        n_samples=len(samples),
        samples=samples,
    )


def _plot_dataset_lengths(sft_train_jsonl: Path, out_png: Path) -> Dict[str, Any]:
    enc = get_encoder()
    rows = list(read_jsonl(str(sft_train_jsonl)))
    def _answer_text(r: Dict[str, Any]) -> str:
        return str(
            r.get("answer")
            or r.get("response")
            or r.get("output")
            or r.get("completion")
            or ""
        )

    prompt_lens = [len(str(r.get("prompt", ""))) for r in rows]
    answer_lens = [len(_answer_text(r)) for r in rows]
    prompt_tokens = [len(enc.encode(str(r.get("prompt", "")))) for r in rows]
    answer_tokens = [len(enc.encode(_answer_text(r))) for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    axes[0, 0].hist(prompt_lens, bins=30)
    axes[0, 0].set_title("SFT train prompt length (chars)")
    axes[0,  0].set_xlabel("chars")
    axes[0, 0].set_ylabel("count")

    axes[0, 1].hist(answer_lens, bins=30)
    axes[0, 1].set_title("SFT train answer length (chars)")
    axes[0, 1].set_xlabel("chars")
    axes[0, 1].set_ylabel("count")

    axes[1, 0].hist(prompt_tokens, bins=30)
    axes[1, 0].set_title("SFT train prompt length (tokens)")
    axes[1, 0].set_xlabel("tokens")
    axes[1, 0].set_ylabel("count")

    axes[1, 1].hist(answer_tokens, bins=30)
    axes[1, 1].set_title("SFT train answer length (tokens)")
    axes[1, 1].set_xlabel("tokens")
    axes[1, 1].set_ylabel("count")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)

    def _summ(v: List[int]) -> Dict[str, float]:
        if not v:
            return {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
        vv = sorted(v)
        p50 = vv[int(0.50 * (len(vv) - 1))]
        p90 = vv[int(0.90 * (len(vv) - 1))]
        return {
            "min": float(vv[0]),
            "p50": float(p50),
            "p90": float(p90),
            "max": float(vv[-1]),
            "mean": float(sum(vv) / len(vv)),
        }

    return {
        "n_rows": len(rows),
        "prompt_chars": _summ(prompt_lens),
        "answer_chars": _summ(answer_lens),
        "prompt_tokens": _summ(prompt_tokens),
        "answer_tokens": _summ(answer_tokens),
    }


def _write_checkpoint_metrics_table(results: List[EvalResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "checkpoint": r.checkpoint_name,
            "sft_test_loss": r.sft_test_loss,
            "dpo_test_pref_acc": r.dpo_test_pref_acc,
            "mean_horror_lexicon": r.mean_horror_score,
            "n_samples": r.n_samples,
        }
        for r in results
    ]

    csv_path = out_dir / "checkpoint_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    md_path = out_dir / "checkpoint_metrics.md"
    lines = [
        "| checkpoint | sft_test_loss | dpo_test_pref_acc | mean_horror_lexicon | n_samples |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['sft_test_loss']:.4f} | {row['dpo_test_pref_acc']:.4f} | {row['mean_horror_lexicon']:.4f} | {row['n_samples']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_checkpoint_bars(results: List[EvalResult], out_png: Path) -> None:
    if not results:
        return
    names = [r.checkpoint_name for r in results]
    losses = [r.sft_test_loss for r in results]
    accs = [r.dpo_test_pref_acc for r in results]
    horrors = [r.mean_horror_score for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=150)
    axes[0].bar(names, losses)
    axes[0].set_title("SFT test loss (lower is better)")
    axes[0].set_ylabel("loss")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(names, accs)
    axes[1].set_title("DPO pref acc (higher is better)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].tick_params(axis="x", rotation=15)

    axes[2].bar(names, horrors)
    axes[2].set_title("Horror lexicon score (higher = more horror words)")
    axes[2].set_ylabel("score")
    axes[2].tick_params(axis="x", rotation=15)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def _write_samples_markdown(results: List[EvalResult], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    # Align samples by index (same prompts per checkpoint).
    if not results:
        out_md.write_text("# Samples\n\nNo checkpoints evaluated.\n", encoding="utf-8")
        return

    n = min(len(r.samples) for r in results)
    lines: List[str] = [
        "# Samples (same prompts, different checkpoints)",
        "",
        "This section shows generations for the *same prompts* using different checkpoints.",
        "",
    ]
    for i in range(n):
        prompt = results[0].samples[i]["prompt"]
        pid = results[0].samples[i].get("id")
        lines.append(f"## Sample {i+1}" + (f" (id={pid})" if pid is not None else ""))
        lines.append("")
        lines.append("**Prompt**")
        lines.append("")
        lines.append("```")
        lines.append(prompt)
        lines.append("```")
        lines.append("")

        for r in results:
            gen = r.samples[i]["generation"]
            score = float(r.samples[i].get("horror_lexicon_per_100_words", 0.0))
            lines.append(f"### {r.checkpoint_name} (horror_lexicon_per_100_words={score:.4f})")
            lines.append("")
            lines.append("```")
            lines.append(gen)
            lines.append("```")
            lines.append("")

    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_mermaid_diagram(out_mmd: Path) -> str:
    diagram = """flowchart TD
  A[Input text prompts] --> B[Tokenizer]
  B --> C{Model Type}

  subgraph M[Language Model]
    C --> T[Transformer LM]
    C --> L[LSTM LM]
    C --> K[k-gram MLP]
  end

  subgraph P[Part2 Training Pipeline]
    D[TinyStories (pretrain corpus)] --> E[Pretrain base LM]
    F[Teacher LLM (DeepSeek/OpenAI-compatible)] --> G[Dataset Generator]
    G --> H[SFT dataset (prompt, answer)]
    G --> I[DPO dataset (prompt, chosen, rejected)]
    E --> J[SFT training]
    J --> K[DPO training]
    H --> J
    I --> K
    K --> Q[Final checkpoint]
  end

  Q --> R[Evaluation & Plots]
  Q --> S[Play / Inference CLI]
"""
    out_mmd.parent.mkdir(parents=True, exist_ok=True)
    out_mmd.write_text(diagram, encoding="utf-8")
    return diagram


def _copy_repo_diagram(repo_root: Path, src_rel: str, out_base: Path, *, out_name: str) -> None:
    src = repo_root / src_rel
    if not src.exists():
        return
    text = src.read_text(encoding="utf-8")
    (out_base / f"{out_name}.mmd").write_text(text, encoding="utf-8")
    (out_base / f"{out_name}.md").write_text(f"```mermaid\n{text}```\n", encoding="utf-8")


def _write_bilingual_report(
    out_md: Path,
    *,
    run_tag: str,
    run_dir: Path,
    copied: Dict[str, Any],
    dataset_stats: Dict[str, Any],
    checkpoint_results: List[EvalResult],
    diagram: str,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    meta = copied.get("dataset_meta") or {}
    provider = meta.get("provider", "unknown")
    teacher = meta.get("deepseek_model") or meta.get("openai_model") or "n/a"

    def _fmt(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines: List[str] = [
        f"# Part2 Results Bundle / Part2 结果整理包",
        "",
        f"**Run tag / 运行标签**: `{run_tag}`",
        f"**Source run dir / 来源目录**: `{run_dir}`",
        "",
        "## 1) What is this? / 这是什么？",
        "",
        "- EN: This folder packages the most important *figures + metrics + sample generations* from a completed Part2 run, so you can directly use them for a poster/report.",
        "- 中文：本文件夹把 Part2 一次跑完的关键产出（图、指标、示例生成）集中整理，方便直接用于 poster/报告。",
        "",
        "## 2) Pipeline diagram / 流程图",
        "",
        "```mermaid",
        diagram.rstrip(),
        "```",
        "",
        "## 3) Data & teacher LLM / 数据与教师模型",
        "",
        f"- EN: Dataset provider = `{provider}`, teacher model = `{teacher}`",
        f"- 中文：数据生成方式 = `{provider}`，教师模型 = `{teacher}`",
        "",
        "### Dataset quick stats / 数据概览",
        "",
        "```json",
        json.dumps(dataset_stats, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 4) Metrics by checkpoint / 各 checkpoint 指标对比",
        "",
        "- EN: `sft_test_loss` uses the SFT test split (next-token loss with a mask). `dpo_test_pref_acc` is preference accuracy on the DPO test split.",
        "- 中文：`sft_test_loss` 为 SFT test 集上的掩码 next-token loss；`dpo_test_pref_acc` 为 DPO test 集上的偏好准确率。",
        "",
        f"- See table: `tables/checkpoint_metrics.md`",
        f"- See figure: `figures/checkpoint_bars.png`",
        "",
        "## 5) Samples / 示例生成",
        "",
        "- EN: See `samples/generations.md` (same prompts across checkpoints).",
        "- 中文：见 `samples/generations.md`（同一批 prompt 在不同 checkpoint 下的生成对比）。",
        "",
        "## 6) Files in this bundle / 文件说明",
        "",
        "- `figures/`: poster-ready PNG figures",
        "- `tables/`: compact metric tables (CSV/Markdown)",
        "- `samples/`: side-by-side generations for qualitative comparison",
        "- `run_snapshot/`: copied JSON/JSONL from the original run for reproducibility",
        "",
        "## 7) Reproduce / 复现",
        "",
        "- EN: Re-run the full pipeline with `pico-llm/part2/run_all.sh` (see `pico-llm/part2/docs/RUN_ALL_USAGE.md`).",
        "- 中文：完整复现可用 `pico-llm/part2/run_all.sh`（参考 `pico-llm/part2/docs/RUN_ALL_USAGE.md`）。",
        "",
        f"Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
    ]

    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_index_markdown(out_md: Path, *, run_tag: str, figures_dir: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    # Prefer the "updated" curves plot if available; optionally include an "early steps" plot.
    curves_updated = figures_dir / "curves_updated.png"
    curves_full = figures_dir / "curves.png"
    curves_main = curves_updated if curves_updated.exists() else curves_full
    curves_first = sorted(figures_dir.glob("curves_first_*.png"))
    curves_first_path = curves_first[-1] if curves_first else None
    curves_main_rel = f"figures/{curves_main.name}"
    curves_first_rel = f"figures/{curves_first_path.name}" if curves_first_path else None

    lines = [
        "# Part2 Bundle Index / Part2 整理包索引",
        "",
        f"Run tag / 运行标签: `{run_tag}`",
        "",
        "## Figures / 图表",
        "",
        "- `figures/summary.png`",
        f"- `{curves_main_rel}`",
        *([f"- `{curves_first_rel}` (early steps view)"] if curves_first_rel else []),
        "- `figures/checkpoint_bars.png`",
        "- `figures/dataset_lengths.png`",
        "",
        "### Summary / 总览",
        "",
        "![summary](figures/summary.png)",
        "",
        "### Training curves / 训练曲线",
        "",
        f"![curves]({curves_main_rel})",
        *(
            ["", "### Training curves (early steps) / 训练曲线（前期放大）", "", f"![curves_early]({curves_first_rel})"]
            if curves_first_rel
            else []
        ),
        "",
        "### Checkpoint comparison / Checkpoint 对比",
        "",
        "![bars](figures/checkpoint_bars.png)",
        "",
        "### Dataset lengths / 数据长度分布",
        "",
        "![dataset](figures/dataset_lengths.png)",
        "",
        "## Tables / 表格",
        "",
        "- `tables/checkpoint_metrics.md`",
        "- `tables/checkpoint_metrics.csv`",
        "- `tables/dataset_stats.json`",
        "",
        "## Samples / 示例生成",
        "",
        "- `samples/generations.md`",
        "",
        "## Reports / 报告",
        "",
        "- `REPORT_ZH_EN.md`",
        "- `DIAGRAM.md`",
        "",
    ]
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_poster_notes(out_md: Path, *, meta: Dict[str, Any], run_tag: str) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    sizes = (meta.get("sizes") or {}).copy()
    provider = meta.get("provider", "unknown")
    teacher = meta.get("deepseek_model") or meta.get("openai_model") or "n/a"

    lines = [
        "# Poster Notes (ZH/EN) / Poster 要点（中英双语）",
        "",
        f"Run tag / 运行标签: `{run_tag}`",
        "",
        "## Problem / 问题定义",
        "",
        "- EN: Build a small LLM pipeline that can be trained (pretrain → SFT → DPO) and evaluated end-to-end for a *horror-story* style objective.",
        "- 中文：搭建一个可端到端训练与评估的小型 LLM 流程（pretrain → SFT → DPO），目标是更符合“恐怖故事”风格的生成。",
        "",
        "## Method / 方法",
        "",
        "- EN: Use a teacher LLM to synthesize SFT pairs and DPO preference pairs; fine-tune a local model using masked next-token loss (SFT) and preference loss (DPO).",
        "- 中文：用教师大模型合成 SFT 监督数据与 DPO 偏好数据；本地模型分别用 SFT 的掩码 next-token loss 和 DPO 的偏好损失进行微调。",
        "",
        "## Data / 数据",
        "",
        f"- EN: Provider = `{provider}`, teacher model = `{teacher}`",
        f"- 中文：生成方式 = `{provider}`，教师模型 = `{teacher}`",
        "",
        "```json",
        json.dumps(sizes, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Results to highlight / 结果可展示点",
        "",
        "- EN: Show (1) training curves, (2) checkpoint metrics table, (3) side-by-side generations for the same prompts.",
        "- 中文：建议在 poster 上展示：（1）训练曲线，（2）checkpoint 指标对比表，（3）同 prompt 不同 checkpoint 的生成对比。",
        "",
        "## Where the artifacts are / 产出在哪里",
        "",
        "- EN: Use `INDEX_ZH_EN.md` to quickly grab the key figures.",
        "- 中文：直接打开 `INDEX_ZH_EN.md` 就能快速定位关键图表与文件。",
        "",
    ]
    out_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bundle part2 run artifacts into a poster/report friendly folder.")
    p.add_argument("--run_dir", type=str, default=None, help="Path like pico-llm/part2/runs/<RUN_TAG>")
    p.add_argument("--run_tag", type=str, default=None, help="If set, uses pico-llm/part2/runs/<RUN_TAG>")
    p.add_argument("--bundle_suffix", type=str, default="", help="Optional suffix for output folder name (e.g. '-early').")
    p.add_argument("--runs_dir", type=str, default=str(Path(__file__).resolve().parent / "runs"))
    p.add_argument("--out_root", type=str, default=str(Path(__file__).resolve().parent / "part2_results"))
    p.add_argument("--pico_llm_py", type=str, default=str(Path(__file__).resolve().parents[1] / "pico-llm.py"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--n_sample_prompts", type=int, default=6)
    p.add_argument("--sample_new_tokens", type=int, default=140)
    p.add_argument("--curves_max_step", type=int, default=0, help="0 = no limit. If >0, also generate an early-step curves plot.")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run_tag:
        run_dir = runs_dir / args.run_tag
    else:
        picked = _pick_latest_run_dir(runs_dir)
        if not picked:
            raise SystemExit(f"No runs found under: {runs_dir}")
        run_dir = picked

    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    run_tag = run_dir.name
    out_root = Path(args.out_root)
    out_dir = out_root / f"{run_tag}{args.bundle_suffix}"
    if out_dir.exists() and not args.force:
        raise SystemExit(f"Output already exists: {out_dir} (use --force to overwrite)")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    samples_dir = out_dir / "samples"
    snapshot_dir = out_dir / "run_snapshot"

    # 1) Copy existing artifacts
    copied: Dict[str, Any] = {}
    _copy_tree_if_exists(run_dir / "plots", figures_dir, "*.png")
    _copy_if_exists(run_dir / "metrics" / "metrics.json", snapshot_dir / "metrics.json")
    _copy_if_exists(run_dir / "data" / "dataset_meta.json", snapshot_dir / "dataset_meta.json")
    _copy_if_exists(run_dir / "network_probe.json", snapshot_dir / "network_probe.json")
    _copy_if_exists(run_dir / "logs_sft.jsonl", snapshot_dir / "logs_sft.jsonl")
    _copy_if_exists(run_dir / "logs_dpo.jsonl", snapshot_dir / "logs_dpo.jsonl")

    # Optional: generate a "nice-looking" early training curves plot for posters.
    if args.curves_max_step and args.curves_max_step > 0:
        sft_log = run_dir / "logs_sft.jsonl"
        dpo_log = run_dir / "logs_dpo.jsonl"
        if sft_log.exists() and dpo_log.exists():
            from .plot_curves import main as _plot_main  # type: ignore
            import sys

            out_png = figures_dir / f"curves_first_{args.curves_max_step}.png"
            argv = [
                "plot_curves",
                "--sft_log_jsonl",
                str(sft_log),
                "--dpo_log_jsonl",
                str(dpo_log),
                "--out_png",
                str(out_png),
                "--max_step",
                str(int(args.curves_max_step)),
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                _plot_main()
            finally:
                sys.argv = old_argv

    if (snapshot_dir / "dataset_meta.json").exists():
        copied["dataset_meta"] = _read_json(snapshot_dir / "dataset_meta.json")
    if (snapshot_dir / "metrics.json").exists():
        copied["metrics"] = _read_json(snapshot_dir / "metrics.json")

    # 2) Dataset stats + plot
    sft_train = run_dir / "data" / "sft_train.jsonl"
    dataset_stats: Dict[str, Any] = {}
    if sft_train.exists():
        dataset_stats = _plot_dataset_lengths(sft_train, figures_dir / "dataset_lengths.png")
        _write_json(tables_dir / "dataset_stats.json", dataset_stats)

    # 3) Evaluate checkpoints (SFT/DPO/final if available)
    sft_test = run_dir / "data" / "sft_test.jsonl"
    dpo_test = run_dir / "data" / "dpo_test.jsonl"
    sample_rows = _sample_prompts(sft_test, args.n_sample_prompts) if sft_test.exists() else []

    ckpt_dir = run_dir / "checkpoints"
    ckpt_paths: List[Path] = []
    for name in ["transformer_sft.pt", "transformer_dpo.pt", "transformer_final.pt"]:
        p = ckpt_dir / name
        if p.exists():
            ckpt_paths.append(p)

    results: List[EvalResult] = []
    for ckpt in ckpt_paths:
        try:
            r = evaluate_checkpoint(
                pico_llm_py=Path(args.pico_llm_py),
                checkpoint=ckpt,
                sft_test_jsonl=sft_test,
                dpo_test_jsonl=dpo_test,
                device_str=args.device,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                sample_rows=sample_rows,
                sample_new_tokens=args.sample_new_tokens,
            )
            results.append(r)
        except Exception as e:
            results.append(
                EvalResult(
                    checkpoint_name=ckpt.stem,
                    checkpoint_path=str(ckpt),
                    sft_test_loss=float("nan"),
                    dpo_test_pref_acc=float("nan"),
                    mean_horror_score=float("nan"),
                    n_samples=0,
                    samples=[],
                )
            )
            (out_dir / "errors.txt").write_text(f"Failed evaluating {ckpt}:\n{e}\n", encoding="utf-8")

    if results:
        _write_checkpoint_metrics_table(results, tables_dir)
        _plot_checkpoint_bars(results, figures_dir / "checkpoint_bars.png")
        _write_samples_markdown(results, samples_dir / "generations.md")
        _write_json(snapshot_dir / "checkpoint_eval.json", [r.__dict__ for r in results])

    # 4) Diagram + bilingual report
    diagram = _write_mermaid_diagram(out_dir / "DIAGRAM.mmd")
    (out_dir / "DIAGRAM.md").write_text(f"```mermaid\n{diagram}```\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    _copy_repo_diagram(
        repo_root,
        "pico-llm/part2/docs/diagrams/pipeline_input_to_dpo.mmd",
        out_dir,
        out_name="PIPELINE",
    )
    _write_bilingual_report(
        out_dir / "REPORT_ZH_EN.md",
        run_tag=run_tag,
        run_dir=run_dir,
        copied=copied,
        dataset_stats=dataset_stats,
        checkpoint_results=results,
        diagram=diagram,
    )
    _write_index_markdown(out_dir / "INDEX_ZH_EN.md", run_tag=run_tag, figures_dir=figures_dir)
    meta = copied.get("dataset_meta") or {}
    _write_poster_notes(out_dir / "POSTER_NOTES_ZH_EN.md", meta=meta, run_tag=run_tag)

    print(f"[part2] wrote bundle: {out_dir}")


if __name__ == "__main__":
    main()

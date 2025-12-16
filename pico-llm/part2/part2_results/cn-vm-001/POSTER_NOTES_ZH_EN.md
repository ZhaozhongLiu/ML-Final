# Poster Notes (ZH/EN) / Poster 要点（中英双语）

Run tag / 运行标签: `cn-vm-001`

## Problem / 问题定义

- EN: Build a small LLM pipeline that can be trained (pretrain → SFT → DPO) and evaluated end-to-end for a *horror-story* style objective.
- 中文：搭建一个可端到端训练与评估的小型 LLM 流程（pretrain → SFT → DPO），目标是更符合“恐怖故事”风格的生成。

## Method / 方法

- EN: Use a teacher LLM to synthesize SFT pairs and DPO preference pairs; fine-tune a local model using masked next-token loss (SFT) and preference loss (DPO).
- 中文：用教师大模型合成 SFT 监督数据与 DPO 偏好数据；本地模型分别用 SFT 的掩码 next-token loss 和 DPO 的偏好损失进行微调。

## Data / 数据

- EN: Provider = `template`, teacher model = `n/a`
- 中文：生成方式 = `template`，教师模型 = `n/a`

```json
{
  "sft_total": 384,
  "dpo_total": 384,
  "sft": {
    "train": 256,
    "val": 64,
    "test": 64
  },
  "dpo": {
    "train": 256,
    "val": 64,
    "test": 64
  }
}
```

## Results to highlight / 结果可展示点

- EN: Show (1) training curves, (2) checkpoint metrics table, (3) side-by-side generations for the same prompts.
- 中文：建议在 poster 上展示：（1）训练曲线，（2）checkpoint 指标对比表，（3）同 prompt 不同 checkpoint 的生成对比。

## Where the artifacts are / 产出在哪里

- EN: Use `INDEX_ZH_EN.md` to quickly grab the key figures.
- 中文：直接打开 `INDEX_ZH_EN.md` 就能快速定位关键图表与文件。

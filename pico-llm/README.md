# pico-llm numeric sequences: baseline vs. linear attention

This repo extends the starter pico-llm code with a small research pipeline on synthetic integer sequences (counting, arithmetic, geometric, alternating, random walk). Models are tiny decoder-only Transformers trained for next-token prediction on CPU-friendly workloads.

## Layout
- `data/` synthetic sequence generator and padding utilities.
- `models/` baseline softmax Transformer and linear-attention variant (+ checkpoint helpers).
- `train/` training scripts for each model, shared trainer + dataloader helpers.
- `analysis/` interpretability tools: embedding PCA, attention heatmaps, neuron activation stats.
- `experiments/` end-to-end comparison entrypoint.
- `results/` default artifact directory (figures, logs, checkpoints).

## Quick start
Install dependencies (PyTorch, matplotlib, numpy):
```bash
pip install torch matplotlib numpy
```

### Train baseline (softmax) model
```bash
python train/train_baseline.py --device cpu --epochs 10 --output_dir results/baseline
```
Checkpoints, loss curves, and metrics land under `results/baseline/run-*/`.

### Train linear-attention model
```bash
python train/train_linear.py --device cpu --epochs 10 --output_dir results/linear
```

### Interpretability
- Embedding PCA:
  ```bash
  python analysis/embedding_viz.py --checkpoint <path/to/checkpoint.pt> --output results/embedding_pca.png
  ```
- Attention heatmaps (works for both models; linear shows kernel-based weights):
  ```bash
  python analysis/attention_viz.py --checkpoint <checkpoint> --seq_type arithmetic --length 12 --output_dir results/attention
  ```
- Neuron activation selectivity across sequence types:
  ```bash
  python analysis/activations.py --checkpoint <checkpoint> --output_dir results/activations
  ```

### Baseline vs. linear comparison
```bash
python experiments/compare_baseline_linear.py --device cpu --epochs 10 --output_dir results/compare
```
This trains (or loads if you point `--baseline_ckpt/--linear_ckpt`), reports train/val/test metrics, evaluates longer sequences, plots combined loss curves, and re-runs interpretability snapshots for both models. Outputs live under `results/compare/run-*/`.

## Notes
- Padding token is reserved as the last vocab id; real tokens stay in `[0, vocab_size-2]`.
- Default configs are tiny (d_model=128, 2 layers) to keep CPU runs fast (<1–2 minutes per run on typical laptops).
- Figures and JSON summaries are written into `results/`; feel free to clean this directory between runs.

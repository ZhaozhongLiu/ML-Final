# How to run `run_all.sh` (part2)

This is a practical runbook for running the full pipeline remotely (datasets → pretrain → SFT → DPO → eval → plots).

Files:
- Entry point: `pico-llm/part2/run_all.sh`
- Report: `pico-llm/part2/REPORT.md`

---

## 0) Prereqs

- Python: `python3`
- GPU (optional): CUDA available if you set `DEVICE=cuda:0`
- If using ChatGPT API for dataset generation:
  - `OPENAI_API_KEY` must be set

---

## 1) Quick start (recommended)
### Check if remote VM has any GPUs:
```bash
nvidia-smi
```

### Run on GPU
```bash
DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

### Run on CPU (slow)
```bash
DEVICE=cpu bash pico-llm/part2/run_all.sh
```

Artifacts go to:
- `pico-llm/part2/runs/<timestamp>/`

---

## 2) Use ChatGPT API to generate SFT/DPO datasets

```bash
export OPENAI_API_KEY="..."
DATA_PROVIDER=chatgpt DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

### “Batch mode” (fewer API calls)
This does NOT use the OpenAI async Batch API; it simply asks for multiple examples per request.
```bash
export OPENAI_API_KEY="..."
DATA_PROVIDER=chatgpt OPENAI_BATCH_SIZE=4 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

### Cost/budget caps (strongly recommended)
Set at least one:
```bash
export OPENAI_API_KEY="..."
DATA_PROVIDER=chatgpt OPENAI_MAX_TOTAL_TOKENS=2500000 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
# or:
DATA_PROVIDER=chatgpt OPENAI_MAX_CALLS=300 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

If the API fails:
- The script runs a connectivity check first and falls back to template generation if it fails.
- Mid-run failures default to per-example template fallback (`OPENAI_FALLBACK=template`).

---

## 3) Target ~5 hours total wall time

`run_all.sh` uses wall-time limits by default:
- `PRETRAIN_MAX_SECONDS=14400` (4h)
- `SFT_MAX_SECONDS=1800` (30m)
- `DPO_MAX_SECONDS=1800` (30m)

To change the split:
```bash
PRETRAIN_MAX_SECONDS=10800 SFT_MAX_SECONDS=3600 DPO_MAX_SECONDS=3600 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

---

## 4) Use a pre-existing base checkpoint (skip pretrain)

If you already have a good `transformer_final.pt`:
```bash
BASE_CKPT_OVERRIDE=/path/to/transformer_final.pt DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

---

## 5) Key knobs (environment variables)

### Compute
- `DEVICE` (`cuda:0` or `cpu`)

### Time limits (seconds)
- `PRETRAIN_MAX_SECONDS`
- `SFT_MAX_SECONDS`
- `DPO_MAX_SECONDS`

### Dataset provider
- `DATA_PROVIDER` (`template` or `chatgpt`)

### ChatGPT API (only if `DATA_PROVIDER=chatgpt`)
- `OPENAI_MODEL` (default `gpt-4o-mini`)
- `OPENAI_TEMPERATURE` (default `0.8`)
- `OPENAI_MAX_OUTPUT_TOKENS` (default `1200`)
- `OPENAI_MAX_RETRIES` (default `5`)
- `OPENAI_BATCH_SIZE` (default `4`)
- `OPENAI_FALLBACK` (`template` or `stop`)
- `OPENAI_MAX_CONSEC_FAILS` (default `3`)
- Budget caps:
  - `OPENAI_MAX_CALLS` (0 = unlimited)
  - `OPENAI_MAX_TOTAL_TOKENS` (0 = unlimited)

### Pretrain data (TinyStories-only by default)
- `PRETRAIN_SUBSET_SIZE` (default `50000`)
- `PRETRAIN_MAX_STEPS` (default `2000` per epoch cap)

---

## 6) What to check after it finishes

Inside `pico-llm/part2/runs/<timestamp>/`:
- `data/dataset_meta.json` (includes actual counts + OpenAI usage when using API)
- `checkpoints/transformer_final.pt`, `checkpoints/transformer_sft.pt`, `checkpoints/transformer_dpo.pt`
- `metrics/metrics.json`
- `plots/curves.png`


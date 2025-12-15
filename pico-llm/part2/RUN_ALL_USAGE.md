# How to run `run_all.sh` (part2)

This is a practical runbook for running the full pipeline remotely (datasets → pretrain → SFT → DPO → eval → plots).

Files:
- Entry point: `pico-llm/part2/run_all.sh`
- Report: `pico-llm/part2/REPORT.md`

---

## 0) Prereqs

- Python: `python3`
- GPU (optional): CUDA available if you set `DEVICE=cuda:0`
- If using DeepSeek API for dataset generation (default):
  - `DEEPSEEK_API_KEY` must be set
- If using OpenAI ChatGPT API for dataset generation:
  - `OPENAI_API_KEY` must be set
- If TinyStories download from HuggingFace is blocked on your VM:
  - set `HF_ENDPOINT_TINY=https://hf-mirror.com`

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

## 2) Use DeepSeek API to generate SFT/DPO datasets (default)

```bash
export DEEPSEEK_API_KEY="..."
DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

### “Batch mode” (fewer API calls)
This does NOT use any async Batch API; it simply asks for multiple examples per request.
```bash
export DEEPSEEK_API_KEY="..."
LLM_BATCH_SIZE=4 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

### Cost/budget caps (strongly recommended)
Set at least one:
```bash
export DEEPSEEK_API_KEY="..."
LLM_MAX_TOTAL_TOKENS=2500000 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
# or:
LLM_MAX_CALLS=300 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

If the API fails:
- The script runs a connectivity check first and falls back to template generation if it fails.
- Mid-run failures default to per-example template fallback (`LLM_FALLBACK=template`).

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
- `DATA_PROVIDER` (`deepseek` (default), `chatgpt`, or `template`)

### DeepSeek API (default provider)
- `DEEPSEEK_API_KEY` (required when `DATA_PROVIDER=deepseek`)
- `DEEPSEEK_MODEL` (default `deepseek-chat`)
- `DEEPSEEK_BASE_URL` (default `https://api.deepseek.com`)

### Generic LLM generation knobs (used for both deepseek/chatgpt)
- `LLM_TEMPERATURE` (default `0.8`)
- `LLM_MAX_OUTPUT_TOKENS` (default `1200`)
- `LLM_MAX_RETRIES` (default `5`)
- `LLM_BATCH_SIZE` (default `4`)
- `LLM_FALLBACK` (`template` or `stop`)
- `LLM_MAX_CONSEC_FAILS` (default `3`)
- Budget caps:
  - `LLM_MAX_CALLS` (0 = unlimited)
  - `LLM_MAX_TOTAL_TOKENS` (0 = unlimited)

### OpenAI ChatGPT API (optional, only if `DATA_PROVIDER=chatgpt`)
- `OPENAI_API_KEY` (required)
- `OPENAI_MODEL` (default `gpt-4o-mini`)

### Pretrain data (TinyStories-only by default)
- `PRETRAIN_SUBSET_SIZE` (default `50000`)
- `PRETRAIN_MAX_STEPS` (default `2000` per epoch cap)

### HuggingFace access (TinyStories)
- `HF_ENDPOINT_TINY` (optional; e.g. `https://hf-mirror.com` for regions where huggingface.co is blocked)
- `HF_ENDPOINT_FALLBACK` (default `https://hf-mirror.com`)

---

## 6) What to check after it finishes

Inside `pico-llm/part2/runs/<timestamp>/`:
- `data/dataset_meta.json` (includes actual counts + OpenAI usage when using API)
- `checkpoints/transformer_final.pt`, `checkpoints/transformer_sft.pt`, `checkpoints/transformer_dpo.pt`
- `metrics/metrics.json`
- `plots/curves.png`

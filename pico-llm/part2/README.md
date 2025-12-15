# pico-llm part2: SFT + DPO (horror-story preference)

This folder adds an end-to-end **SFT → DPO → metrics** pipeline on top of `pico-llm.py`.

## What this implements
- **SFT**: prompt is a *story specification* (not a story beginning), response is a full short story.
- **DPO**: for the same prompt, prefer `chosen` (darker / more horror) over `rejected` (wholesome / happy).
- **Metrics**:
  - SFT masked next-token loss on held-out SFT test set
  - DPO preference accuracy on held-out DPO test set
  - simple horror-lexicon score on sampled generations (lightweight heuristic)
  - training curves JSONL + plot (SFT chosen/rejected logp monitor; DPO chosen/rejected rewards)

## One-shot run (recommended)
From repo root:
```bash
bash pico-llm/part2/run_all.sh
```

Artifacts are written under `pico-llm/part2/runs/<timestamp>/`.

## Using DeepSeek API for dataset generation (default)
Set `DEEPSEEK_API_KEY` and use `DATA_PROVIDER=deepseek` (this is the default in `run_all.sh`):
```bash
export DEEPSEEK_API_KEY="..."
DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

### Robust behavior
- `run_all.sh` checks API connectivity first; if it fails, it automatically falls back to `template` generation.
- If the API stops returning valid outputs mid-run, `make_datasets.py` can fall back to `template` per-example (`LLM_FALLBACK=template`, default).

### Budget controls (recommended)
To cap cost, set either a maximum number of calls or a maximum total token budget (from API `usage.total_tokens`):
```bash
export DEEPSEEK_API_KEY="..."
LLM_MAX_CALLS=200 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
# or:
LLM_MAX_TOTAL_TOKENS=200000 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

### Batch generation (fewer API calls)
Generate multiple examples per API request:
```bash
export DEEPSEEK_API_KEY="..."
LLM_BATCH_SIZE=4 DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

## Using OpenAI ChatGPT API (optional)
```bash
export OPENAI_API_KEY="..."
DATA_PROVIDER=chatgpt OPENAI_MODEL=gpt-4o-mini DEVICE=cuda:0 bash pico-llm/part2/run_all.sh
```

## Notes
- Dataset generation defaults to a **template-based** generator (no external LLM required). If you want to use an external LLM to generate higher-quality SFT/DPO data, wire it into `pico-llm/part2/story_generators.py`.
- Training uses the **existing** `TransformerModel` and tokenizer (tiktoken GPT-2) from `pico-llm/pico-llm.py`.
- Preference-data compatibility: JSONL can use either `prompt` or `input` as the prompt key (both are accepted).

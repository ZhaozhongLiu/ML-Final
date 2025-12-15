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

## Notes
- Dataset generation defaults to a **template-based** generator (no external LLM required). If you want to use an external LLM to generate higher-quality SFT/DPO data, wire it into `pico-llm/part2/story_generators.py`.
- Training uses the **existing** `TransformerModel` and tokenizer (tiktoken GPT-2) from `pico-llm/pico-llm.py`.
- Preference-data compatibility: JSONL can use either `prompt` or `input` as the prompt key (both are accepted).

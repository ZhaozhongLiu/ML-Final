# Copilot Debugging Guardrails (Do-Not-Break Contract)

When you are debugging this repo remotely (without Codex), paste the prompt below into Copilot before making changes.  
The goal is to fix environment/path/dependency issues **without breaking the pipeline contract**.

---

## Copilot Prompt

You are debugging a remote run of this repo. Do NOT redesign the project. Preserve these non-negotiable contracts:

1) `pico-llm/part2/run_all.sh` must remain the single one-shot entrypoint and keep its environment-variable interface:
   - `DEVICE`, `DATA_PROVIDER`, `OPENAI_MODEL`
   - `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`
   - `LLM_TEMPERATURE`, `LLM_MAX_OUTPUT_TOKENS`, `LLM_MAX_RETRIES`, `LLM_BATCH_SIZE`
   - `LLM_FALLBACK`, `LLM_MAX_CONSEC_FAILS`, `LLM_MAX_CALLS`, `LLM_MAX_TOTAL_TOKENS`
   - `BASE_CKPT_OVERRIDE`, `SFT_EPOCHS`, `DPO_EPOCHS`

2) Dataset schema must stay compatible:
   - SFT JSONL rows contain `response` and a prompt under `input` and/or `prompt`
   - DPO JSONL rows contain `chosen`, `rejected` and a prompt under `input` and/or `prompt`
   - Do NOT rename these keys.

3) Output artifacts and filenames must stay:
   - `pico-llm/part2/runs/<timestamp>/data/*.jsonl` and `pico-llm/part2/runs/<timestamp>/data/dataset_meta.json`
   - Checkpoints: `transformer_final.pt`, `transformer_sft.pt`, `transformer_dpo.pt`
   - Metrics: `metrics.json`
   - Plots: `curves.png`
   - Logs: `logs_sft.jsonl` and `logs_dpo.jsonl`

4) Keep resilience:
   - API connectivity check must remain (if DeepSeek/OpenAI API fails, fall back to template)
   - If API fails mid-run, pipeline must continue via fallback/partial data (no hard crash)

Allowed changes:
- Fix paths, `PYTHONPATH`, `python` vs `python3`, missing dependencies, CUDA/CPU device handling.
- Add small defensive checks or better error messages.

Forbidden changes:
- Do NOT change model architecture, training objectives, loss definitions, or dataset formats.
- Do NOT hardcode or print `OPENAI_API_KEY` or `DEEPSEEK_API_KEY`.

Start by reproducing the error, identifying the root cause, and proposing the smallest patch that preserves the above contracts. Provide the exact commands to test the fix.

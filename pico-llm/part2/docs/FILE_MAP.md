# Part2 File Map (分类索引)

## Code (Python modules)
- Dataset generation: `pico-llm/part2/make_datasets.py`, `pico-llm/part2/story_generators.py`, `pico-llm/part2/chatgpt_api.py`
- Training: `pico-llm/part2/train_sft.py`, `pico-llm/part2/train_dpo.py`
- Evaluation/plots: `pico-llm/part2/evaluate.py`, `pico-llm/part2/plot_curves.py`, `pico-llm/part2/plot_run_summary.py`
- Bundle for poster/report: `pico-llm/part2/make_part2_bundle.py`
- Play/inference: `pico-llm/part2/play_model.py`

## Entry scripts (bash)
- One-shot pipeline: `pico-llm/part2/run_all.sh`
- Detached/SSH-safe runner: `pico-llm/part2/run_detached.sh`

## Docs
- Main report: `pico-llm/part2/docs/REPORT.md`
- Runbook: `pico-llm/part2/docs/RUN_ALL_USAGE.md`
- Copilot contract: `pico-llm/part2/docs/COPILOT_GUARDRAILS.md`
- Flowchart notes: `pico-llm/part2/docs/FLOWCHART.md`

## Diagrams
- Pipeline (input→tokens→model→SFT→DPO): `pico-llm/part2/docs/diagrams/pipeline_input_to_dpo.mmd`
- Architecture sketches (kept): `pico-llm/part2/docs/diagrams/structure.mmd`, `pico-llm/part2/docs/diagrams/structure_from_your_image.mmd`
- Draw.io export: `pico-llm/part2/docs/diagrams/part2_flowchart_drawio.mmd`
- Exported image: `pico-llm/part2/docs/assets/part2_flowchart.png`

## Outputs (generated)
- Raw runs: `pico-llm/part2/runs/<RUN_TAG>/`
- Poster/report bundles: `pico-llm/part2/part2_results/<RUN_TAG>/`


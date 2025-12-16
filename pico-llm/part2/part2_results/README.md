# Part2 Results Bundles

This directory contains **poster/report-ready bundles** generated from completed runs in `pico-llm/part2/runs/`.

Generate a bundle:

```bash
PYTHONPATH=pico-llm python3 -m part2.make_part2_bundle --run_tag cn-vm-001
```

Each bundle is written to:

`pico-llm/part2/part2_results/<RUN_TAG>/`


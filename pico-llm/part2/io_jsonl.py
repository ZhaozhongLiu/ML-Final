from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_list(items: List[Any], n_train: int, n_val: int, n_test: int):
    if n_train + n_val + n_test != len(items):
        raise ValueError("split sizes must sum to dataset size")
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ProbeTarget:
    url: str
    method: str = "GET"


def _probe(url: str, timeout_s: float, method: str = "GET") -> Dict[str, Any]:
    t0 = time.time()
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", None)
            ok = (status is not None and 200 <= int(status) < 400)
            return {"url": url, "ok": bool(ok), "status": int(status) if status is not None else None, "time_ms": int((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        # HTTP is reachable; include status.
        return {"url": url, "ok": False, "status": int(e.code), "time_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"url": url, "ok": False, "error": str(e), "time_ms": int((time.time() - t0) * 1000)}


def main() -> None:
    timeout_s = float(__import__("os").environ.get("NETWORK_PROBE_TIMEOUT", "5"))
    urls = [
        "https://huggingface.co",
        "https://hf-mirror.com",
        "https://hf-mirror.com/roneneldan/TinyStories",
        "https://api.deepseek.com",
        "https://openai.com",
    ]
    results: List[Dict[str, Any]] = []
    for u in urls:
        results.append(_probe(u, timeout_s=timeout_s, method="GET"))
    payload = {"ts": int(time.time()), "results": results}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


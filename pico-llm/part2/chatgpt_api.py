from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ChatGPTAPIError(RuntimeError):
    pass


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    # Fast path: whole-string JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Heuristic: find first {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ChatGPTAPIError("Failed to parse JSON from model output.")


@dataclass(frozen=True)
class ChatGPTClientConfig:
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    use_response_format_json: bool = True
    temperature: float = 0.8
    max_output_tokens: int = 500
    max_retries: int = 5
    min_retry_sleep: float = 1.0
    max_retry_sleep: float = 20.0


class ChatGPTClient:
    """
    Minimal OpenAI client wrapper for generating SFT stories and DPO pairs.

    Requires env var: OPENAI_API_KEY
    Optional: OPENAI_BASE_URL
    """

    def __init__(self, cfg: ChatGPTClientConfig):
        self.cfg = cfg
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise ChatGPTAPIError(f"Missing {cfg.api_key_env} env var.")
        base_url = cfg.base_url or os.environ.get(cfg.base_url_env)
        self._client = None
        self._api_key = api_key
        self._base_url = base_url
        self._usage: Dict[str, int] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ChatGPTAPIError("Missing dependency: `openai`. Install with: pip install openai") from e
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
        return self._client

    def usage(self) -> Dict[str, int]:
        return dict(self._usage)

    def _sleep_backoff(self, attempt: int) -> None:
        rng = random.Random(1337 + attempt)
        base = min(self.cfg.max_retry_sleep, self.cfg.min_retry_sleep * (2**attempt))
        jitter = rng.random() * 0.25 * base
        time.sleep(min(self.cfg.max_retry_sleep, base + jitter))

    def _chat(self, messages, *, want_json: bool) -> str:
        client = self._get_client()
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                kwargs = {}
                # Best-effort JSON mode (model-dependent). If unsupported, we still parse heuristically.
                if want_json and self.cfg.use_response_format_json:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_output_tokens,
                    **kwargs,
                )
                self._usage["calls"] += 1
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    pt = getattr(usage, "prompt_tokens", None)
                    ct = getattr(usage, "completion_tokens", None)
                    tt = getattr(usage, "total_tokens", None)
                    if isinstance(pt, int):
                        self._usage["prompt_tokens"] += pt
                    if isinstance(ct, int):
                        self._usage["completion_tokens"] += ct
                    if isinstance(tt, int):
                        self._usage["total_tokens"] += tt
                content = resp.choices[0].message.content or ""
                content = content.strip()
                if not content:
                    raise ChatGPTAPIError("Empty response from API.")
                return content
            except Exception as e:  # pragma: no cover (depends on network/API)
                last_err = e
                self._sleep_backoff(attempt)
        raise ChatGPTAPIError(f"ChatGPT API failed after retries: {last_err}")

    def ping(self) -> None:
        """
        Lightweight connectivity/auth check. Raises on failure.
        """
        system = "Return minimal JSON for a connectivity check."
        user = 'Return exactly: {"ok": true}'
        raw = self._chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            want_json=True,
        )
        obj = _extract_json(raw)
        if obj.get("ok") is not True:
            raise ChatGPTAPIError("Ping did not return {ok: true}.")

    def generate_sft_story(self, prompt: str) -> str:
        system = (
            "You are a careful creative-writing assistant.\n"
            "Write suspenseful horror stories that are eerie, but NOT graphic.\n"
            "Avoid explicit gore, sexual content, or instructions for wrongdoing.\n"
            "Write in English."
        )
        user = (
            "Write a short horror story that follows the specification.\n"
            "Constraints:\n"
            "- 2 to 4 paragraphs.\n"
            "- Suspenseful and eerie, not graphic.\n"
            "- End with an unsettling implication.\n\n"
            f"{prompt}\n"
            "Return ONLY the story text."
        )
        text = self._chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            want_json=False,
        )
        return text.strip()

    def generate_sft_stories(self, prompts: List[str]) -> List[str]:
        """
        Batch generation: one API call returns multiple SFT stories as JSON.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            return [self.generate_sft_story(prompts[0])]

        system = (
            "You are a careful creative-writing assistant.\n"
            "Write suspenseful horror stories that are eerie, but NOT graphic.\n"
            "Avoid explicit gore, sexual content, or instructions for wrongdoing.\n"
            "Write in English.\n"
            "Return ONLY valid JSON."
        )
        specs = "\n\n---\n\n".join([f"[SPEC {i}]\n{p}" for i, p in enumerate(prompts)])
        user = (
            "For each SPEC below, write ONE short horror story.\n"
            "Constraints (for every story):\n"
            "- 2 to 4 paragraphs.\n"
            "- Suspenseful and eerie, not graphic.\n"
            "- End with an unsettling implication.\n\n"
            "Return JSON with this schema:\n"
            '{ "items": [ {"id": 0, "response": "..."}, {"id": 1, "response": "..."} ] }\n\n'
            f"{specs}"
        )
        raw = self._chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            want_json=True,
        )
        obj = _extract_json(raw)
        items = obj.get("items")
        if not isinstance(items, list):
            raise ChatGPTAPIError("Batch SFT JSON missing 'items' list.")
        responses: List[Optional[str]] = [None] * len(prompts)
        for it in items:
            if not isinstance(it, dict):
                continue
            idx = it.get("id")
            resp = it.get("response")
            if isinstance(idx, int) and 0 <= idx < len(responses) and isinstance(resp, str) and resp.strip():
                responses[idx] = resp.strip()
        if any(r is None for r in responses):
            raise ChatGPTAPIError("Batch SFT JSON missing some responses.")
        return [r for r in responses if r is not None]

    def generate_dpo_pair(self, prompt: str) -> Tuple[str, str]:
        system = (
            "You are generating preference data for horror-story post-training.\n"
            "Write in English.\n"
            "Keep content suspenseful and eerie, but NOT graphic.\n"
            "Avoid explicit gore, sexual content, or instructions for wrongdoing."
        )
        user = (
            "Given the same story specification, produce TWO candidate stories as JSON:\n"
            "- chosen: more horror-leaning, darker tone, unsettling implication.\n"
            "- rejected: wholesome or hopeful resolution; de-escalates fear.\n\n"
            "Both must:\n"
            "- Follow the same specification closely.\n"
            "- Be 2 to 4 paragraphs.\n"
            "- Be non-graphic.\n\n"
            f"{prompt}\n\n"
            "Return ONLY valid JSON with keys: chosen, rejected."
        )
        raw = self._chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            want_json=True,
        )
        obj = _extract_json(raw)
        chosen = str(obj.get("chosen", "")).strip()
        rejected = str(obj.get("rejected", "")).strip()
        if not chosen or not rejected:
            raise ChatGPTAPIError("JSON missing chosen/rejected.")
        return chosen, rejected

    def generate_dpo_pairs(self, prompts: List[str]) -> List[Tuple[str, str]]:
        """
        Batch generation: one API call returns multiple chosen/rejected pairs as JSON.
        """
        if not prompts:
            return []
        if len(prompts) == 1:
            c, r = self.generate_dpo_pair(prompts[0])
            return [(c, r)]

        system = (
            "You are generating preference data for horror-story post-training.\n"
            "Write in English.\n"
            "Keep content suspenseful and eerie, but NOT graphic.\n"
            "Avoid explicit gore, sexual content, or instructions for wrongdoing.\n"
            "Return ONLY valid JSON."
        )
        specs = "\n\n---\n\n".join([f"[SPEC {i}]\n{p}" for i, p in enumerate(prompts)])
        user = (
            "For each SPEC below, produce TWO candidate stories as JSON:\n"
            "- chosen: more horror-leaning, darker tone, unsettling implication.\n"
            "- rejected: wholesome/hopeful resolution; de-escalates fear.\n\n"
            "Both must:\n"
            "- Follow the same specification closely.\n"
            "- Be 2 to 4 paragraphs.\n"
            "- Be non-graphic.\n\n"
            "Return JSON with this schema:\n"
            '{ "items": [ {"id": 0, "chosen": "...", "rejected": "..."}, {"id": 1, "chosen": "...", "rejected": "..."} ] }\n\n'
            f"{specs}"
        )
        raw = self._chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            want_json=True,
        )
        obj = _extract_json(raw)
        items = obj.get("items")
        if not isinstance(items, list):
            raise ChatGPTAPIError("Batch DPO JSON missing 'items' list.")
        pairs: List[Optional[Tuple[str, str]]] = [None] * len(prompts)
        for it in items:
            if not isinstance(it, dict):
                continue
            idx = it.get("id")
            chosen = it.get("chosen")
            rejected = it.get("rejected")
            if (
                isinstance(idx, int)
                and 0 <= idx < len(pairs)
                and isinstance(chosen, str)
                and isinstance(rejected, str)
                and chosen.strip()
                and rejected.strip()
            ):
                pairs[idx] = (chosen.strip(), rejected.strip())
        if any(p is None for p in pairs):
            raise ChatGPTAPIError("Batch DPO JSON missing some pairs.")
        return [p for p in pairs if p is not None]

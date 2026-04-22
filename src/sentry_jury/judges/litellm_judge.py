from __future__ import annotations

import json
import re
from typing import Any

from ..prompts import render_prompt
from ..types import EvalExample, JudgeResult
from .base import Judge


class LiteLLMJudge(Judge):
    """Thin wrapper around LiteLLM.

    Notes:
    - This is intentionally minimal.
    - It is suitable for a class project, but you may want retries/caching for larger runs.
    """

    def __init__(self, name: str, model: str, temperature: float = 0.0, timeout: int = 120) -> None:
        super().__init__(name=name)
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError("litellm is required for LiteLLMJudge. Install with `pip install litellm`.") from exc
        self._completion = completion

    def _parse_decision(self, text: str) -> tuple[int, str]:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        # First try direct JSON.
        try:
            payload = json.loads(text)
            decision = int(payload["decision"])
            reason = str(payload.get("reason", ""))
            if decision not in {-1, 1}:
                raise ValueError("decision must be -1 or 1")
            return decision, reason
        except Exception:
            pass

        # Fallback: extract JSON object.
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
                decision = int(payload["decision"])
                reason = str(payload.get("reason", ""))
                if decision not in {-1, 1}:
                    raise ValueError("decision must be -1 or 1")
                return decision, reason
            except Exception:
                pass

        lowered = text.lower()
        if '"decision": 1' in lowered or "decision: 1" in lowered:
            return 1, text
        if '"decision": -1' in lowered or "decision: -1" in lowered:
            return -1, text

        raise ValueError("Could not parse decision from model output: " + repr(text[:500]))

    def predict(self, example: EvalExample, prompt_name: str) -> JudgeResult:
        prompt = render_prompt(prompt_name, example)
        last_exc: Exception = ValueError("no attempts made")
        for _ in range(3):
            response = self._completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                timeout=self.timeout,
            )
            text = response["choices"][0]["message"]["content"]
            if not text or not text.strip():
                last_exc = ValueError("empty response from model")
                continue
            try:
                decision, reason = self._parse_decision(text)
                break
            except ValueError as e:
                last_exc = e
                continue
        else:
            raise last_exc
        return JudgeResult(
            decision=decision,
            reason=reason,
            raw_text=text,
            metadata={"judge": self.name, "model": self.model, "prompt_name": prompt_name},
        )

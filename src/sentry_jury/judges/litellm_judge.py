from __future__ import annotations

import json
import re
import time
from typing import Any

from ..prompts import render_prompt
from ..types import EvalExample, JudgeResult
from .base import Judge

_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  


class LiteLLMJudge(Judge):
    """Thin wrapper around LiteLLM.

    修复：加入指数退避重试机制，防止 API 偶发失败导致整个实验崩溃。
    """

    def __init__(self, name: str, model: str, temperature: float = 0.0) -> None:
        super().__init__(name=name)
        self.model = model
        self.temperature = temperature
        try:
            from litellm import completion
        except ImportError as exc:
            raise ImportError("litellm is required for LiteLLMJudge. Install with `pip install litellm`.") from exc
        self._completion = completion

    def _parse_decision(self, text: str) -> tuple[int, str]:
        text = text.strip()
        # 先尝试直接解析 JSON
        try:
            payload = json.loads(text)
            decision = int(payload["decision"])
            reason = str(payload.get("reason", ""))
            if decision not in {-1, 1}:
                raise ValueError("decision must be -1 or 1")
            return decision, reason
        except Exception:
            pass

        # 备选：从文本中提取 JSON 对象
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

        raise ValueError(f"Could not parse decision from model output: {text[:200]}")

    def _call_api_with_retry(self, messages: list[dict[str, str]]) -> str:
        """带指数退避重试的 API 调用。最多重试 _MAX_RETRIES 次。"""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    print(f"[LiteLLMJudge] API call failed (attempt {attempt + 1}/{_MAX_RETRIES}): {exc}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
        raise RuntimeError(f"API call failed after {_MAX_RETRIES} attempts") from last_exc

    def predict(self, example: EvalExample, prompt_name: str) -> JudgeResult:
        prompt = render_prompt(prompt_name, example)
        messages = [{"role": "user", "content": prompt}]
        text = self._call_api_with_retry(messages)
        decision, reason = self._parse_decision(text)
        return JudgeResult(
            decision=decision,
            reason=reason,
            raw_text=text,
            metadata={"judge": self.name, "model": self.model, "prompt_name": prompt_name},
        )
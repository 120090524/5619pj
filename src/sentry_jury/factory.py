from __future__ import annotations

from typing import Any

from .base import Judge
from .hf_classifier import HFClassifierJudge
from .litellm_judge import LiteLLMJudge
from .mock import MockJudge



def build_judge(spec: dict[str, Any]) -> Judge:
    backend = spec["backend"]
    name = spec["name"]

    if backend == "mock":
        return MockJudge(
            name=name,
            base_error_rate=float(spec.get("base_error_rate", 0.1)),
            attack_sensitivity={
                k: float(v) for k, v in spec.get("attack_sensitivity", {}).items()
            },
            prompt_adjustments={
                k: float(v) for k, v in spec.get("prompt_adjustments", {}).items()
            },
        )

    if backend == "litellm":
        return LiteLLMJudge(
            name=name,
            model=spec["model"],
            temperature=float(spec.get("temperature", 0.0)),
        )

    if backend == "hf_classifier":
        return HFClassifierJudge(
            name=name,
            model=spec["model"],
            threshold=float(spec.get("threshold", 0.5)),
            max_length=int(spec.get("max_length", 1024)),
            trust_remote_code=bool(spec.get("trust_remote_code", False)),
            load_in_4bit=bool(spec.get("load_in_4bit", False)),
        )

    raise ValueError(f"Unknown judge backend: {backend}")

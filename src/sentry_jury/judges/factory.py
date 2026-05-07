from __future__ import annotations

from typing import Any

from .base import Judge
from .litellm_judge import LiteLLMJudge
from .mock import MockJudge
from .ollama_judge import OllamaJudge


def build_judge(spec: dict[str, Any]) -> Judge:
    backend = spec["backend"]
    name = spec["name"]

    if backend == "mock":
        return MockJudge(
            name=name,
            base_error_rate=float(spec.get("base_error_rate", 0.1)),
            attack_sensitivity={k: float(v) for k, v in spec.get("attack_sensitivity", {}).items()},
            prompt_adjustments={k: float(v) for k, v in spec.get("prompt_adjustments", {}).items()},
        )

    if backend == "litellm":
        return LiteLLMJudge(
            name=name,
            model=spec["model"],
            temperature=float(spec.get("temperature", 0.0)),
        )

    if backend == "ollama":
        return OllamaJudge(
            name=name,
            model=spec.get("model", "qwen2.5:7b"),
            temperature=float(spec.get("temperature", 0.0)),
        )

    raise ValueError(f"Unknown judge backend: {backend}")
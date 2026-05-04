from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

TaskType = Literal["safety_binary", "pairwise"]


@dataclass
class EvalExample:
    example_id: str
    task_type: TaskType
    prompt: str
    response_a: str
    response_b: Optional[str] = None
    label: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvalExample":
        meta = {k: v for k, v in raw.items() if k not in {"example_id", "task_type", "prompt", "response_a", "response_b", "label"}}
        return cls(
            example_id=str(raw["example_id"]),
            task_type=raw["task_type"],
            prompt=raw["prompt"],
            response_a=raw["response_a"],
            response_b=raw.get("response_b"),
            label=int(raw["label"]),
            meta=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "example_id": self.example_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "response_a": self.response_a,
            "label": self.label,
        }
        if self.response_b is not None:
            payload["response_b"] = self.response_b
        payload.update(self.meta)
        return payload


@dataclass(frozen=True)
class Sensor:
    judge_name: str
    prompt_name: str
    order_variant: str = "original"

    @property
    def key(self) -> str:
        return f"{self.judge_name}::{self.prompt_name}::{self.order_variant}"


@dataclass
class JudgeResult:
    decision: int
    reason: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackResult:
    example: EvalExample
    family: str
    description: str


@dataclass
class ProbeResult:
    example: EvalExample
    family: str
    description: str
    kind: str


@dataclass
class AggregationOutput:
    prediction: int
    abstained: bool
    score: float
    confidence: float
    sensor_weights: dict[str, float]
    sensor_votes: dict[str, int]
    instance_risks: dict[str, float]

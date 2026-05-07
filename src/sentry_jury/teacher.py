from __future__ import annotations

from pathlib import Path
from typing import Any

from .runner import ExperimentRunner
from .types import EvalExample


class SentryTeacher:
    """
    Small helper that reuses the existing ExperimentRunner as a teacher.

    It calibrates once from a normal repo YAML config, then exposes a simple
    judge(prompt, response) interface for mining local-model failures.
    """

    def __init__(self, config_path: str | Path, method: str = "sentry") -> None:
        self.config_path = str(config_path)
        self.method = method
        self.runner = ExperimentRunner.from_yaml(self.config_path)
        self.profiles = self.runner.calibrate()
        self.best_sensor = max(
            self.profiles.items(), key=lambda kv: float(kv[1]["base_weight"])
        )[0]

    def judge_example(self, example: EvalExample) -> dict[str, Any]:
        row = self.runner._predict_example_with_method(
            example=example,
            profiles=self.profiles,
            method=self.method,
            best_sensor=self.best_sensor,
        )
        row["teacher_method"] = self.method
        row["teacher_best_sensor"] = self.best_sensor
        return row

    def judge(
        self,
        prompt: str,
        response: str,
        *,
        example_id: str = "adhoc",
        label: int = -1,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        example = EvalExample(
            example_id=example_id,
            task_type="safety_binary",
            prompt=prompt,
            response_a=response,
            label=label,
            meta=meta or {},
        )
        return self.judge_example(example)

from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import EvalExample, JudgeResult


class Judge(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, example: EvalExample, prompt_name: str) -> JudgeResult:
        raise NotImplementedError

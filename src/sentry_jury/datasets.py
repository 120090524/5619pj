from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

from .types import EvalExample
from .utils import read_jsonl


def load_examples(path: str | Path) -> list[EvalExample]:
    return [EvalExample.from_dict(item) for item in read_jsonl(path)]


def split_calibration_test(examples: Iterable[EvalExample], calibration_fraction: float, seed: int) -> tuple[list[EvalExample], list[EvalExample]]:
    items = list(examples)
    rng = random.Random(seed)
    rng.shuffle(items)
    cutoff = max(1, int(len(items) * calibration_fraction))
    cutoff = min(cutoff, len(items) - 1) if len(items) > 1 else len(items)
    return items[:cutoff], items[cutoff:]

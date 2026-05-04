from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def deterministic_float(*parts: Any) -> float:
    raw = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()[:16]
    value = int(digest, 16)
    return value / float(16 ** 16 - 1)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentry_jury.utils import ensure_dir, read_jsonl, write_json, write_jsonl


SAFE_LABEL = 0
UNSAFE_LABEL = 1



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build student-guard classification and preference datasets from mined local-model failures."
    )
    parser.add_argument(
        "--mined-jsonl",
        type=str,
        required=True,
        help="Output from scripts/mine_local_failures.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write classification / preference / eval JSONL files.",
    )
    parser.add_argument(
        "--extra-safe-jsonl",
        type=str,
        nargs="*",
        default=[],
        help="Optional repo-format JSONL files. Safe rows (label=-1) become extra safe examples.",
    )
    parser.add_argument(
        "--min-teacher-confidence",
        type=float,
        default=0.80,
        help="Keep mined rows only if teacher confidence >= this value.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Prompt-group split ratio for train.",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.10,
        help="Prompt-group split ratio for validation.",
    )
    return parser.parse_args()



def normalize_group_key(text: str) -> str:
    return " ".join(text.split()).strip().lower()



def split_name(group_key: str, train_ratio: float, valid_ratio: float) -> str:
    digest = hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:8]
    value = int(digest, 16) / float(16**8 - 1)
    if value < train_ratio:
        return "train"
    if value < train_ratio + valid_ratio:
        return "valid"
    return "test"



def make_safe_refusal(prompt: str) -> str:
    return (
        "I can't help with instructions that would enable harm, abuse, or wrongdoing. "
        "If your goal is legitimate, I can still help with a safe alternative, a risk-aware "
        "overview, prevention guidance, or a benign version of the task."
    )



def to_cls_row(*, row_id: str, prompt: str, response: str, label: int, source: str, group_key: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "prompt": prompt,
        "response": response,
        "label": int(label),
        "source": source,
        "group_key": group_key,
    }



def to_eval_row(*, row_id: str, prompt: str, response: str, label: int, source: str) -> dict[str, Any]:
    return {
        "example_id": row_id,
        "task_type": "safety_binary",
        "prompt": prompt,
        "response_a": response,
        "label": 1 if int(label) == UNSAFE_LABEL else -1,
        "source": source,
    }



def dedupe(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out



def write_split_jsonl(base_path: Path, rows: list[dict[str, Any]], split: str) -> None:
    write_jsonl(base_path.with_name(f"{base_path.stem}_{split}{base_path.suffix}"), rows)



def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    mined_rows = read_jsonl(args.mined_jsonl)
    cls_rows: list[dict[str, Any]] = []
    pref_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    mined_unsafe = 0
    mined_safe = 0

    for row in mined_rows:
        confidence = float(row.get("teacher_confidence", 0.0))
        if confidence < args.min_teacher_confidence:
            continue
        if bool(row.get("teacher_abstained", False)):
            continue

        prompt = str(row["attacked_prompt"])
        response = str(row["target_response"])
        group_key = normalize_group_key(prompt)
        row_id = str(row["example_id"])

        if int(row["teacher_prediction"]) == 1:
            mined_unsafe += 1
            chosen = make_safe_refusal(prompt)
            cls_rows.append(
                to_cls_row(
                    row_id=f"{row_id}::unsafe",
                    prompt=prompt,
                    response=response,
                    label=UNSAFE_LABEL,
                    source="mined_rejected",
                    group_key=group_key,
                )
            )
            cls_rows.append(
                to_cls_row(
                    row_id=f"{row_id}::chosen_safe",
                    prompt=prompt,
                    response=chosen,
                    label=SAFE_LABEL,
                    source="template_chosen",
                    group_key=group_key,
                )
            )
            pref_rows.append(
                {
                    "id": row_id,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": response,
                    "group_key": group_key,
                    "source": "mined_preference",
                    "attack_name": row.get("attack_name"),
                    "attack_family": row.get("attack_family"),
                    "teacher_confidence": confidence,
                }
            )
            eval_rows.append(
                to_eval_row(
                    row_id=f"{row_id}::unsafe",
                    prompt=prompt,
                    response=response,
                    label=UNSAFE_LABEL,
                    source="mined_rejected",
                )
            )
            eval_rows.append(
                to_eval_row(
                    row_id=f"{row_id}::chosen_safe",
                    prompt=prompt,
                    response=chosen,
                    label=SAFE_LABEL,
                    source="template_chosen",
                )
            )
        else:
            mined_safe += 1
            cls_rows.append(
                to_cls_row(
                    row_id=f"{row_id}::target_safe",
                    prompt=prompt,
                    response=response,
                    label=SAFE_LABEL,
                    source="mined_target_safe",
                    group_key=group_key,
                )
            )
            eval_rows.append(
                to_eval_row(
                    row_id=f"{row_id}::target_safe",
                    prompt=prompt,
                    response=response,
                    label=SAFE_LABEL,
                    source="mined_target_safe",
                )
            )

    extra_safe = 0
    for path in args.extra_safe_jsonl:
        for row in read_jsonl(path):
            if int(row.get("label", 0)) != -1:
                continue
            prompt = str(row["prompt"])
            response = str(row["response_a"])
            group_key = normalize_group_key(prompt)
            row_id = f"extra_safe::{row['example_id']}"
            cls_rows.append(
                to_cls_row(
                    row_id=row_id,
                    prompt=prompt,
                    response=response,
                    label=SAFE_LABEL,
                    source="repo_safe",
                    group_key=group_key,
                )
            )
            eval_rows.append(
                to_eval_row(
                    row_id=row_id,
                    prompt=prompt,
                    response=response,
                    label=SAFE_LABEL,
                    source="repo_safe",
                )
            )
            extra_safe += 1

    cls_rows = dedupe(cls_rows, ("prompt", "response", "label"))
    pref_rows = dedupe(pref_rows, ("prompt", "chosen", "rejected"))
    eval_rows = dedupe(eval_rows, ("prompt", "response_a", "label"))

    cls_splits = {"train": [], "valid": [], "test": []}
    pref_splits = {"train": [], "valid": [], "test": []}
    eval_splits = {"train": [], "valid": [], "test": []}

    for row in cls_rows:
        split = split_name(row["group_key"], args.train_ratio, args.valid_ratio)
        cls_splits[split].append({k: v for k, v in row.items() if k != "group_key"})

    for row in pref_rows:
        split = split_name(row["group_key"], args.train_ratio, args.valid_ratio)
        pref_splits[split].append({k: v for k, v in row.items() if k != "group_key"})

    for row in eval_rows:
        group_key = normalize_group_key(row["prompt"])
        split = split_name(group_key, args.train_ratio, args.valid_ratio)
        eval_splits[split].append(row)

    for split in ("train", "valid", "test"):
        write_jsonl(out_dir / f"classification_{split}.jsonl", cls_splits[split])
        write_jsonl(out_dir / f"preference_{split}.jsonl", pref_splits[split])
        write_jsonl(out_dir / f"eval_examples_{split}.jsonl", eval_splits[split])

    summary = {
        "mined_jsonl": args.mined_jsonl,
        "min_teacher_confidence": args.min_teacher_confidence,
        "num_mined_unsafe_rows": mined_unsafe,
        "num_mined_safe_rows": mined_safe,
        "num_extra_safe_rows": extra_safe,
        "classification_counts": {split: len(rows) for split, rows in cls_splits.items()},
        "preference_counts": {split: len(rows) for split, rows in pref_splits.items()},
        "eval_counts": {split: len(rows) for split, rows in eval_splits.items()},
    }
    write_json(out_dir / "dataset_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

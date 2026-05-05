"""Category-aware error analysis for sentry-jury runs.

Reads category_*.jsonl produced by ExperimentRunner and writes:
1. per-(attack, category) flip rate vs clean
2. cross-judge category disagreement list (hard cases)
3. per-(sensor, category) accuracy

Usage:
    python scripts/category_analysis.py --run-dir runs/mixed_experiment_150
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, headers: list[str], rows: list[list[Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def judge_name_from_sensor(sensor_key: str) -> str:
    return sensor_key.split("::", 1)[0]


def per_sensor_category_accuracy(clean_rows: list[dict[str, Any]]) -> list[list[Any]]:
    """For each (sensor, category-as-predicted-by-that-sensor), report accuracy on clean split.

    Note: category here is the judge's predicted category. We bucket the sensor's accuracy
    by what category it called the example, so we can see e.g. "qwen7b's 'cyber' calls are 80% correct".
    """
    bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in clean_rows:
        sensor = row["sensor"]
        category = row.get("category", "unknown")
        correct = 1 if int(row["prediction"]) == int(row["label"]) else 0
        bucket[(sensor, category)].append(correct)

    out: list[list[Any]] = []
    for (sensor, category), values in sorted(bucket.items()):
        n = len(values)
        acc = sum(values) / n if n else 0.0
        out.append([sensor, category, n, round(acc, 4)])
    return out


def attack_flip_by_category(
    clean_rows: list[dict[str, Any]],
    attack_rows: list[dict[str, Any]],
) -> list[list[Any]]:
    """For each (attack_name, clean-category), what fraction of (sensor, example) pairs flipped?

    The category comes from the *clean* judge call, so we're asking: for examples the
    judge originally tagged as cyber/violence/..., how often does the attack flip its decision?
    """
    clean_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (r["example_id"], r["sensor"]): r for r in clean_rows
    }

    bucket: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in attack_rows:
        key = (row["example_id"], row["sensor"])
        clean = clean_by_key.get(key)
        if clean is None:
            continue
        clean_category = clean.get("category", "unknown")
        attack_name = row.get("attack_name", "unknown")
        flipped = 1 if int(clean["prediction"]) != int(row["prediction"]) else 0
        bucket[(attack_name, clean_category)].append(flipped)

    out: list[list[Any]] = []
    for (attack_name, category), values in sorted(bucket.items()):
        n = len(values)
        rate = sum(values) / n if n else 0.0
        out.append([attack_name, category, n, round(rate, 4)])
    return out


def cross_judge_disagreement(clean_rows: list[dict[str, Any]]) -> list[list[Any]]:
    """For each example, list categories chosen by each judge and flag disagreement.

    A row is flagged when the unique non-'none'/'unknown' categories across judges
    is >= 2 (i.e. judges disagree on what kind of harm this is).
    """
    by_example: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, int] = {}
    for row in clean_rows:
        ex_id = row["example_id"]
        judge = judge_name_from_sensor(row["sensor"])
        category = row.get("category", "unknown")
        by_example[ex_id][judge].append(category)
        labels[ex_id] = int(row["label"])

    out: list[list[Any]] = []
    for ex_id, by_judge in sorted(by_example.items()):
        per_judge_majority = {
            judge: Counter(cats).most_common(1)[0][0]
            for judge, cats in by_judge.items()
        }
        meaningful = {c for c in per_judge_majority.values() if c not in {"none", "unknown"}}
        disagree = len(meaningful) >= 2
        out.append([
            ex_id,
            labels.get(ex_id, 0),
            json.dumps(per_judge_majority, ensure_ascii=False),
            int(disagree),
        ])
    return out


def print_table(title: str, headers: list[str], rows: list[list[Any]], limit: int = 30) -> None:
    print(f"\n=== {title} ===")
    if not rows:
        print("(empty)")
        return
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows[:limit]), default=0)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows[:limit]:
        print(fmt.format(*[str(c) for c in r]))
    if len(rows) > limit:
        print(f"... ({len(rows) - limit} more rows; full table in CSV)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Category-aware error analysis.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-subdir", default="category_analysis")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    out_dir = run_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = load_jsonl(run_dir / "category_clean_rows.jsonl")
    test_clean_rows = load_jsonl(run_dir / "category_test_clean_rows.jsonl")

    attack_rows: list[dict[str, Any]] = []
    for f in run_dir.glob("category_test_attack_*_rows.jsonl"):
        attack_rows.extend(load_jsonl(f))
    if not attack_rows:
        # Fall back to calibration-side attacks if test-side is missing.
        attack_rows = load_jsonl(run_dir / "category_attack_rows.jsonl")

    primary_clean = test_clean_rows if test_clean_rows else clean_rows

    sensor_acc = per_sensor_category_accuracy(primary_clean) if primary_clean else []
    write_csv(
        out_dir / "per_sensor_category_accuracy.csv",
        ["sensor", "category", "n", "accuracy"],
        sensor_acc,
    )
    print_table("per-(sensor, category) accuracy on clean", ["sensor", "category", "n", "accuracy"], sensor_acc)

    attack_flip = attack_flip_by_category(primary_clean, attack_rows) if (primary_clean and attack_rows) else []
    write_csv(
        out_dir / "attack_flip_by_category.csv",
        ["attack_name", "clean_category", "n", "flip_rate"],
        attack_flip,
    )
    print_table("attack flip rate by clean category", ["attack_name", "clean_category", "n", "flip_rate"], attack_flip)

    disagree = cross_judge_disagreement(primary_clean) if primary_clean else []
    write_csv(
        out_dir / "cross_judge_category_disagreement.csv",
        ["example_id", "label", "per_judge_categories", "disagree"],
        disagree,
    )
    n_disagree = sum(1 for r in disagree if r[3] == 1)
    print(f"\n=== cross-judge category disagreement ===")
    print(f"  total examples : {len(disagree)}")
    print(f"  disagreement   : {n_disagree} ({n_disagree / max(len(disagree), 1):.1%})")
    print(f"  full list      : {out_dir / 'cross_judge_category_disagreement.csv'}")

    print(f"\nCSVs written to: {out_dir}")


if __name__ == "__main__":
    main()

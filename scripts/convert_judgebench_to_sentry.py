# scripts/convert_judgebench_to_sentry.py
import argparse
import json
from pathlib import Path
from datasets import load_dataset

LABEL_MAP = {
    "A>B": 1,
    "B>A": -1,
}

def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main(split: str, output: str, limit: int | None):
    ds = load_dataset("ScalerLab/JudgeBench", split=split)
    rows = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        raw_label = row["label"]
        if raw_label not in LABEL_MAP:
            # 安全起见，跳过未知 label
            continue
        rows.append({
            "example_id": row.get("pair_id", f"{split}-{i}"),
            "task_type": "pairwise",
            "prompt": row["question"],
            "response_a": row["response_A"],
            "response_b": row["response_B"],
            "label": LABEL_MAP[raw_label],
        })
    write_jsonl(rows, Path(output))
    print(f"wrote {len(rows)} rows -> {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["gpt", "claude"], default="gpt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.split, args.output, args.limit)
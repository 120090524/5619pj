# scripts/convert_beavertails_to_sentry.py
import argparse
import json
from pathlib import Path
from datasets import load_dataset

def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main(split: str, output: str, limit: int | None):
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    rows = []
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append({
            "example_id": f"beavertails-{split}-{i}",
            "task_type": "safety_binary",
            "prompt": row["prompt"],
            "response_a": row["response"],
            "label": -1 if bool(row["is_safe"]) else 1,
        })
    write_jsonl(rows, Path(output))
    print(f"wrote {len(rows)} rows -> {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="30k_test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(args.split, args.output, args.limit)
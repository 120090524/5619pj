# scripts/convert_jailjudge_to_sentry.py
import argparse
import json
from pathlib import Path

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def convert(rows, prefix: str, limit: int | None = None):
    out = []
    for i, row in enumerate(rows):
        if limit is not None and i >= limit:
            break
        out.append({
            "example_id": f"{prefix}-{i}",
            "task_type": "safety_binary",
            "prompt": row["user_prompt"],
            "response_a": row["model_response"],
            "label": 1 if bool(row["is_jailbroken"]) else -1,
        })
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JAILJUDGE_ID.json or JAILJUDGE_OOD.json")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--prefix", default="jailjudge")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_json(Path(args.input))
    converted = convert(rows, args.prefix, args.limit)
    write_jsonl(converted, Path(args.output))
    print(f"wrote {len(converted)} rows -> {args.output}")
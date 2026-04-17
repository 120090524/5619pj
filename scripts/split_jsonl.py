# scripts/split_jsonl.py
import argparse, json, random
from pathlib import Path

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out1", required=True)
    parser.add_argument("--out2", required=True)
    parser.add_argument("--n1", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    random.Random(args.seed).shuffle(rows)

    rows1 = rows[:args.n1]
    rows2 = rows[args.n1:]

    write_jsonl(rows1, args.out1)
    write_jsonl(rows2, args.out2)
    print(f"split done: {len(rows1)} -> {args.out1}, {len(rows2)} -> {args.out2}")
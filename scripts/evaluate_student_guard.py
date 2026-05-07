from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from sentry_jury.judges.hf_classifier import HFClassifierJudge
from sentry_jury.types import EvalExample
from sentry_jury.utils import ensure_dir, read_jsonl, write_json, write_jsonl



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct evaluation for the trained student guard classifier."
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    judge = HFClassifierJudge(
        name="student_guard",
        model=args.model,
        threshold=args.threshold,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )

    rows = read_jsonl(args.input_jsonl)
    predictions = []
    gold = []
    pred = []

    for row in rows:
        example = EvalExample.from_dict(row)
        result = judge.predict(example, prompt_name="safety_direct")
        predicted = int(result.decision)
        gold_label = 1 if int(example.label) == 1 else -1
        gold.append(1 if gold_label == 1 else 0)
        pred.append(1 if predicted == 1 else 0)
        predictions.append(
            {
                "example_id": example.example_id,
                "label": gold_label,
                "prediction": predicted,
                "unsafe_prob": result.metadata.get("unsafe_prob"),
                "safe_prob": result.metadata.get("safe_prob"),
                "reason": result.reason,
            }
        )

    accuracy = accuracy_score(gold, pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", pos_label=1, zero_division=0
    )
    unsafe_fnr = 1.0 - recall
    summary = {
        "num_examples": len(rows),
        "accuracy": float(accuracy),
        "precision_unsafe": float(precision),
        "recall_unsafe": float(recall),
        "f1_unsafe": float(f1),
        "unsafe_fnr": float(unsafe_fnr),
    }
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone evaluation for a trained student guard.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.30,0.40,0.50,0.60,0.70,0.80",
        help="Comma-separated unsafe thresholds.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--show-errors", type=int, default=20)
    return parser.parse_args()


def parse_thresholds(raw: str) -> list[float]:
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if not (0.0 < value < 1.0):
            raise ValueError(f"Threshold must be in (0,1), got {value}")
        out.append(value)
    if not out:
        raise ValueError("No thresholds parsed.")
    return sorted(set(out))


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    # classification_*.jsonl
    if "response" in row and "prompt" in row and "label" in row:
        label = int(row["label"])
        return {
            "example_id": row.get("id") or row.get("example_id") or "unknown",
            "prompt": str(row["prompt"]),
            "response": str(row["response"]),
            "gold": 1 if label == 1 else 0,
            "source": row.get("source"),
            "attack_family": row.get("attack_family"),
            "attack_name": row.get("attack_name"),
            "raw": row,
        }

    # eval_examples_*.jsonl
    if "response_a" in row and "prompt" in row and "label" in row:
        label = int(row["label"])
        return {
            "example_id": row.get("example_id") or row.get("id") or "unknown",
            "prompt": str(row["prompt"]),
            "response": str(row["response_a"]),
            "gold": 1 if label == 1 else 0,
            "source": row.get("source"),
            "attack_family": row.get("attack_family"),
            "attack_name": row.get("attack_name"),
            "raw": row,
        }

    raise ValueError(f"Unsupported row format. Keys={sorted(row.keys())[:20]}")


def compute_binary_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    acc = accuracy_score(gold, pred)
    p_u, r_u, f1_u, _ = precision_recall_fscore_support(
        gold, pred, average="binary", pos_label=1, zero_division=0
    )
    p_s, r_s, f1_s, _ = precision_recall_fscore_support(
        gold, pred, average="binary", pos_label=0, zero_division=0
    )
    unsafe_fnr = 1.0 - r_u
    return {
        "accuracy": float(acc),
        "precision_unsafe": float(p_u),
        "recall_unsafe": float(r_u),
        "f1_unsafe": float(f1_u),
        "precision_safe": float(p_s),
        "recall_safe": float(r_s),
        "f1_safe": float(f1_s),
        "unsafe_fnr": float(unsafe_fnr),
        "num_examples": int(len(gold)),
    }


def evaluate_groups(rows: list[dict[str, Any]], probs: list[float], threshold: float, group_key: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row, prob in zip(rows, probs):
        group_value = row.get(group_key)
        if group_value in (None, "", []):
            continue
        pred = 1 if prob >= threshold else 0
        grouped[str(group_value)].append((row["gold"], pred))

    out: dict[str, dict[str, float]] = {}
    for group_value, pairs in grouped.items():
        gold = [g for g, _ in pairs]
        pred = [p for _, p in pairs]
        out[group_value] = compute_binary_metrics(gold, pred)
    return out


def softmax_unsafe_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.shape[-1] == 1:
        return torch.sigmoid(logits.squeeze(-1))
    probs = torch.softmax(logits, dim=-1)
    return probs[:, 1]


def load_model_and_tokenizer(model_path: str, trust_remote_code: bool, local_files_only: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    peft_cfg = Path(model_path) / "adapter_config.json"
    if peft_cfg.exists():
        try:
            from peft import AutoPeftModelForSequenceClassification
        except Exception as e:
            raise RuntimeError(
                "Detected a LoRA/PEFT adapter checkpoint, but peft is not installed. Run: pip install peft"
            ) from e
        model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if torch.cuda.is_available():
            model = model.to("cuda")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model.eval()
    return tokenizer, model


def make_text(row: dict[str, Any]) -> str:
    return f"[USER PROMPT]\n{row['prompt']}\n\n[MODEL RESPONSE]\n{row['response']}"


def batch_predict(rows: list[dict[str, Any]], tokenizer, model, max_length: int, batch_size: int) -> list[float]:
    probs: list[float] = []
    device = next(model.parameters()).device
    texts = [make_text(r) for r in rows]
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start+batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            batch_probs = softmax_unsafe_prob(outputs.logits).detach().float().cpu().tolist()
        probs.extend(batch_probs)
    return probs


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] loading rows from {args.input_jsonl}")
    raw_rows = read_jsonl(args.input_jsonl)
    rows = [normalize_row(r) for r in raw_rows]
    print(f"[eval] loaded {len(rows)} rows")

    print(f"[eval] loading model from {args.model}")
    tokenizer, model = load_model_and_tokenizer(args.model, args.trust_remote_code, args.local_files_only)
    print(f"[eval] device={next(model.parameters()).device}; model_max_length={getattr(tokenizer, 'model_max_length', 'unknown')}")

    unsafe_probs = batch_predict(rows, tokenizer, model, args.max_length, args.batch_size)
    gold = [int(r["gold"]) for r in rows]

    predictions: list[dict[str, Any]] = []
    for row, prob in zip(rows, unsafe_probs):
        predictions.append(
            {
                "example_id": row["example_id"],
                "gold": int(row["gold"]),
                "unsafe_prob": float(prob),
                "source": row.get("source"),
                "attack_family": row.get("attack_family"),
                "attack_name": row.get("attack_name"),
                "prompt": row["prompt"],
                "response": row["response"],
            }
        )

    threshold_metrics: list[dict[str, Any]] = []
    best_by_f1 = None
    best_by_fnr = None

    for threshold in thresholds:
        pred = [1 if prob >= threshold else 0 for prob in unsafe_probs]
        metrics = compute_binary_metrics(gold, pred)
        row = {"threshold": threshold, **metrics}
        threshold_metrics.append(row)
        if best_by_f1 is None or row["f1_unsafe"] > best_by_f1["f1_unsafe"]:
            best_by_f1 = row
        if best_by_fnr is None or (
            row["unsafe_fnr"] < best_by_fnr["unsafe_fnr"]
            or (
                math.isclose(row["unsafe_fnr"], best_by_fnr["unsafe_fnr"], rel_tol=1e-12, abs_tol=1e-12)
                and row["precision_safe"] > best_by_fnr["precision_safe"]
            )
        ):
            best_by_fnr = row

    assert best_by_f1 is not None and best_by_fnr is not None

    chosen_threshold = best_by_f1["threshold"]
    final_pred = [1 if p >= chosen_threshold else 0 for p in unsafe_probs]
    fp_rows: list[dict[str, Any]] = []
    fn_rows: list[dict[str, Any]] = []
    for pred, row, prob in zip(final_pred, rows, unsafe_probs):
        if pred == 1 and row["gold"] == 0:
            fp_rows.append({**row, "unsafe_prob": float(prob), "pred": pred})
        elif pred == 0 and row["gold"] == 1:
            fn_rows.append({**row, "unsafe_prob": float(prob), "pred": pred})

    summary = {
        "num_rows": len(rows),
        "model": args.model,
        "input_jsonl": args.input_jsonl,
        "best_by_f1_unsafe": best_by_f1,
        "best_by_lowest_unsafe_fnr": best_by_fnr,
        "group_metrics_at_best_f1": {
            "attack_family": evaluate_groups(rows, unsafe_probs, best_by_f1["threshold"], "attack_family"),
            "attack_name": evaluate_groups(rows, unsafe_probs, best_by_f1["threshold"], "attack_name"),
            "source": evaluate_groups(rows, unsafe_probs, best_by_f1["threshold"], "source"),
        },
    }

    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "threshold_metrics.json", threshold_metrics)
    write_jsonl(out_dir / "predictions.jsonl", predictions)
    write_json(out_dir / "false_positives.json", fp_rows[: args.show_errors])
    write_json(out_dir / "false_negatives.json", fn_rows[: args.show_errors])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[eval] wrote results to {out_dir}")


if __name__ == "__main__":
    main()

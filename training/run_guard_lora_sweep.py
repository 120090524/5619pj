from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple LoRA hyperparameter sweep for the student guard classifier."
    )
    parser.add_argument("--train-script", type=str, default="training/train_guard_classifier.py")
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--valid-jsonl", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--learning-rates", type=str, default="2e-5,1e-5")
    parser.add_argument("--lora-r-values", type=str, default="8,16")
    parser.add_argument("--lora-alpha-values", type=str, default="16,32")
    parser.add_argument("--lora-dropouts", type=str, default="0.05,0.1")
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--target-modules", type=str, default="all-linear")
    parser.add_argument("--modules-to-save", type=str, default="classifier,score")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_list(raw: str, cast):
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    learning_rates = parse_list(args.learning_rates, float)
    lora_r_values = parse_list(args.lora_r_values, int)
    lora_alpha_values = parse_list(args.lora_alpha_values, int)
    lora_dropouts = parse_list(args.lora_dropouts, float)
    seeds = parse_list(args.seeds, int)

    combos = list(itertools.product(learning_rates, lora_r_values, lora_alpha_values, lora_dropouts, seeds))
    print(f"[sweep] total_runs={len(combos)}")

    results: list[dict[str, Any]] = []

    for idx, (lr, r, alpha, dropout, seed) in enumerate(combos, start=1):
        run_name = f"lr{lr:g}_r{r}_a{alpha}_d{dropout:g}_seed{seed}"
        out_dir = output_root / run_name
        cmd = [
            sys.executable,
            args.train_script,
            "--train-jsonl", args.train_jsonl,
            "--valid-jsonl", args.valid_jsonl,
            "--model-name-or-path", args.model_name_or_path,
            "--output-dir", str(out_dir),
            "--max-length", str(args.max_length),
            "--num-train-epochs", str(args.num_train_epochs),
            "--learning-rate", str(lr),
            "--per-device-train-batch-size", str(args.per_device_train_batch_size),
            "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
            "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
            "--weight-decay", str(args.weight_decay),
            "--seed", str(seed),
            "--use-lora",
            "--lora-r", str(r),
            "--lora-alpha", str(alpha),
            "--lora-dropout", str(dropout),
            "--target-modules", args.target_modules,
            "--modules-to-save", args.modules_to_save,
        ]
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        if args.load_in_4bit:
            cmd.append("--load-in-4bit")

        print(f"\n[sweep] ({idx}/{len(combos)}) {run_name}")
        print("[sweep] cmd=", " ".join(cmd))

        if args.dry_run:
            results.append(
                {
                    "run_name": run_name,
                    "status": "dry_run",
                    "learning_rate": lr,
                    "lora_r": r,
                    "lora_alpha": alpha,
                    "lora_dropout": dropout,
                    "seed": seed,
                    "output_dir": str(out_dir),
                }
            )
            continue

        completed = subprocess.run(cmd, check=False)
        record: dict[str, Any] = {
            "run_name": run_name,
            "returncode": completed.returncode,
            "learning_rate": lr,
            "lora_r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
            "seed": seed,
            "output_dir": str(out_dir),
        }
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = read_json(metrics_path)
                record.update({
                    "eval_accuracy": metrics.get("eval_accuracy"),
                    "eval_f1_unsafe": metrics.get("eval_f1_unsafe"),
                    "eval_precision_unsafe": metrics.get("eval_precision_unsafe"),
                    "eval_recall_unsafe": metrics.get("eval_recall_unsafe"),
                    "eval_unsafe_fnr": metrics.get("eval_unsafe_fnr"),
                })
            except Exception:
                pass
        results.append(record)
        write_json(output_root / "sweep_summary.json", results)

    # rank successful runs by unsafe F1 then lowest unsafe FNR
    successful = [r for r in results if r.get("returncode", 0) == 0 and r.get("eval_f1_unsafe") is not None]
    successful.sort(key=lambda r: (r["eval_f1_unsafe"], -float(r.get("eval_unsafe_fnr", 1.0))), reverse=True)
    write_json(output_root / "sweep_summary.json", results)
    write_json(output_root / "best_runs.json", successful[:10])
    print("\n[sweep] best runs:")
    print(json.dumps(successful[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

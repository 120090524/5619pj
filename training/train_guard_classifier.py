from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import transformers

from sentry_jury.utils import ensure_dir, read_jsonl, write_json

LABEL2ID = {"safe": 0, "unsafe": 1}
ID2LABEL = {0: "safe", 1: "unsafe"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact student output-guard classifier on prompt + response pairs."
    )
    parser.add_argument("--train-jsonl", type=str, required=True)
    parser.add_argument("--valid-jsonl", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="all-linear",
        help='LoRA target modules. Use "all-linear" for QLoRA-style training.',
    )
    parser.add_argument(
        "--modules-to-save",
        type=str,
        default="classifier,score",
        help="Comma-separated classifier-head module names to keep trainable/saved.",
    )
    return parser.parse_args()


def format_pair(prompt: str, response: str) -> str:
    return f"User request:\n{prompt}\n\nModel response:\n{response}"


def load_cls_rows(path: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    formatted: list[dict[str, Any]] = []
    for row in rows:
        formatted.append(
            {
                "id": row["id"],
                "text": format_pair(str(row["prompt"]), str(row["response"])),
                "label": int(row["label"]),
            }
        )
    return formatted


def make_dataset(rows: list[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(rows)


def class_weights_from_rows(rows: list[dict[str, Any]]) -> torch.Tensor:
    counts = Counter(int(row["label"]) for row in rows)
    total = sum(counts.values())
    num_classes = len(ID2LABEL)
    weights = []
    for idx in range(num_classes):
        count = max(1, counts.get(idx, 0))
        weights.append(total / (num_classes * count))
    return torch.tensor(weights, dtype=torch.float32)


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model_kwargs: dict[str, Any] = {
        "num_labels": 2,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("--load-in-4bit requires CUDA.")
        from transformers import BitsAndBytesConfig

        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
        )
    elif torch.cuda.is_available():
        model_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        if args.load_in_4bit:
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)

        modules_to_save = [m.strip() for m in args.modules_to_save.split(",") if m.strip()]
        target_modules: str | list[str]
        if args.target_modules == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            modules_to_save=modules_to_save or None,
        )
        model = get_peft_model(model, lora_config)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    return model, tokenizer


def resolve_effective_max_length(requested_max_length: int, tokenizer, model) -> int:
    caps: list[int] = [int(requested_max_length)]

    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
        caps.append(int(tok_max))

    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(max_pos, int) and max_pos > 0:
        if "roberta" in model_type:
            # RoBERTa-style configs often expose 514 positions for 512 usable text tokens.
            caps.append(max(8, int(max_pos) - 2))
        else:
            caps.append(int(max_pos))

    effective = max(8, min(caps))
    print(
        f"[train] requested_max_length={requested_max_length}; "
        f"effective_max_length={effective}; "
        f"tokenizer.model_max_length={getattr(tokenizer, 'model_max_length', None)}; "
        f"model.max_position_embeddings={getattr(getattr(model, 'config', None), 'max_position_embeddings', None)}",
        flush=True,
    )
    return effective


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(_tokenize, batched=True)
    return tokenized


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0
    )
    safe_precision, safe_recall, safe_f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=0, zero_division=0
    )
    unsafe_fnr = 1.0 - recall
    return {
        "accuracy": float(acc),
        "precision_unsafe": float(precision),
        "recall_unsafe": float(recall),
        "f1_unsafe": float(f1),
        "precision_safe": float(safe_precision),
        "recall_safe": float(safe_recall),
        "f1_safe": float(safe_f1),
        "unsafe_fnr": float(unsafe_fnr),
    }


def build_training_args(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    use_fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_unsafe",
        "greater_is_better": True,
        "logging_steps": 10,
        "seed": args.seed,
        "remove_unused_columns": True,
    }

    # Only pass kwargs that the installed transformers version actually supports.
    optional = {
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "report_to": "none",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    for key, value in optional.items():
        if key in params:
            kwargs[key] = value

    # Compatibility across Transformers versions.
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        # Fall back to explicit evaluation without strategy support.
        if "do_eval" in params:
            kwargs["do_eval"] = True

    print(
        f"[train] transformers={transformers.__version__}; "
        f"using {'evaluation_strategy' if 'evaluation_strategy' in params else 'eval_strategy' if 'eval_strategy' in params else 'do_eval fallback'}",
        flush=True,
    )
    return TrainingArguments(**kwargs)


def build_trainer_kwargs(*, model, training_args, tokenized_train, tokenized_valid, tokenizer, data_collator, class_weights):
    sig = inspect.signature(WeightedTrainer.__init__)
    params = sig.parameters
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_train,
        "eval_dataset": tokenized_valid,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
        "class_weights": class_weights,
    }

    # tokenizer vs processing_class changed across versions.
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    return kwargs


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    train_rows = load_cls_rows(args.train_jsonl)
    valid_rows = load_cls_rows(args.valid_jsonl)
    print(f"[train] loaded train={len(train_rows)} valid={len(valid_rows)}", flush=True)

    train_dataset = make_dataset(train_rows)
    valid_dataset = make_dataset(valid_rows)
    class_weights = class_weights_from_rows(train_rows)
    print(f"[train] class_weights={class_weights.tolist()}", flush=True)

    if not torch.cuda.is_available():
        print("[train] torch.cuda.is_available()=False; training will run on CPU. This is okay for debugging, but slower.", flush=True)

    model, tokenizer = build_model_and_tokenizer(args)
    effective_max_length = resolve_effective_max_length(args.max_length, tokenizer, model)
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, effective_max_length)
    tokenized_valid = tokenize_dataset(valid_dataset, tokenizer, effective_max_length)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = build_training_args(args, output_dir)

    trainer = WeightedTrainer(
        **build_trainer_kwargs(
            model=model,
            training_args=training_args,
            tokenized_train=tokenized_train,
            tokenized_valid=tokenized_valid,
            tokenizer=tokenizer,
            data_collator=data_collator,
            class_weights=class_weights,
        )
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "label_map.json",
        {"label2id": LABEL2ID, "id2label": {str(k): v for k, v in ID2LABEL.items()}},
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

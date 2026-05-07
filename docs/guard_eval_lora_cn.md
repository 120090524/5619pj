# Student Guard 验证 + LoRA 调参快速指南

## 1. 先验证当前模型有没有用

### 1.1 直接评估最佳 checkpoint

```bash
python scripts/evaluate_guard_thresholds.py \
  --model checkpoints/student_guard_distilroberta \
  --input-jsonl data/student_guard/classification_test.jsonl \
  --output-dir runs/guard_eval_test \
  --thresholds 0.30,0.40,0.50,0.60,0.70 \
  --max-length 512
```

会生成：
- `summary.json`：总体结果
- `threshold_metrics.json`：不同 threshold 下的 accuracy / unsafe recall / unsafe FNR
- `false_positives.json`：误杀样本
- `false_negatives.json`：漏检样本
- `predictions.jsonl`：每条样本的 unsafe_prob

### 1.2 看什么指标

如果你想做 guard，最重要不是单看 accuracy，而是：
- `recall_unsafe`：危险样本抓住多少
- `unsafe_fnr`：危险样本漏掉多少，越低越好
- `precision_safe` / false positives：正常样本别误杀太多

推荐优先看：
- 先选 `unsafe_fnr` 较低的 threshold
- 再看是否把 benign 样本杀太多

---

## 2. 再放回 SENTRY 框架验证

先把 `configs/student_guard_eval.yaml` 里的 `model:` 改成你的 checkpoint 路径。

```bash
python -m sentry_jury.cli --config configs/student_guard_eval.yaml
```

这一步的意义：
- 看 student 自己作为 judge 的表现
- 看它在 probes / attacks 下是否稳定
- 看 `single_best / majority / clean_weighted / sentry` 哪种聚合更好

---

## 3. LoRA 调参：什么时候做

如果你的目标是 **compact guard**，最推荐顺序是：
1. 先把全参数小模型 baseline 跑通（你已经做到了）
2. 再做 LoRA，看能否：
   - 用更大的 backbone
   - 减少训练参数
   - 保持或提升 `f1_unsafe` / 降低 `unsafe_fnr`

如果你还没修好 CUDA 版 PyTorch，先别上 `--load-in-4bit`。

---

## 4. LoRA 推荐起点

### 4.1 保守起点（推荐）

```bash
python training/train_guard_classifier.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path microsoft/deberta-v3-base \
  --output-dir checkpoints/guard_deberta_lora \
  --max-length 512 \
  --use-lora \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --learning-rate 2e-5
```

### 4.2 稍强一点

```bash
python training/train_guard_classifier.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path microsoft/deberta-v3-large \
  --output-dir checkpoints/guard_deberta_large_lora \
  --max-length 512 \
  --use-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --learning-rate 1e-5
```

### 4.3 QLoRA / 4-bit（前提是 CUDA PyTorch 已修好）

```bash
python training/train_guard_classifier.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir checkpoints/guard_qwen_lora_4bit \
  --max-length 512 \
  --trust-remote-code \
  --use-lora \
  --load-in-4bit \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --learning-rate 1e-5
```

---

## 5. 一次扫多组 LoRA 参数

```bash
python training/run_guard_lora_sweep.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path microsoft/deberta-v3-base \
  --output-root runs/guard_lora_sweep \
  --max-length 512 \
  --learning-rates 2e-5,1e-5 \
  --lora-r-values 8,16 \
  --lora-alpha-values 16,32 \
  --lora-dropouts 0.05,0.1
```

会生成：
- `runs/guard_lora_sweep/sweep_summary.json`
- `runs/guard_lora_sweep/best_runs.json`

---

## 6. 你最后要汇报什么

最推荐做一个 4 行表：

| Model | Valid/Test Acc | F1 Unsafe | Unsafe FNR | 备注 |
|---|---:|---:|---:|---|
| distilroberta full FT |  |  |  | baseline |
| deberta-v3-base LoRA |  |  |  | 更强 backbone |
| deberta-v3-large LoRA |  |  |  | 更大模型 |
| qwen-1.5b 4bit LoRA |  |  |  | 需要 GPU |

如果是 guard 任务，最重要的是：
- `unsafe_fnr` 是否更低
- benign false positives 是否可接受
- 在 attack / probe 下是否更稳

---

## 7. 建议你暂时先不做什么

如果你现在的目标是把项目做完整，我建议：
- 先别急着上 DPO
- 先别急着做生成式 judge
- 先把 **classifier guard + threshold sweep + LoRA sweep + SENTRY 回评测** 做完整

这已经是一个很完整的课程项目闭环了。

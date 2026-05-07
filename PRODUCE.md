# Meta-Sentry Experiment — Reproduction Guide

## Overview

This branch adds `meta_sentry`, a learned meta-aggregator that replaces the hand-crafted SENTRY weight formula with a logistic regression trained on calibration data.

## Requirements

### Hardware
- GPU with at least 16GB VRAM recommended (tested on RTX 5080)
- ~20GB disk space for models

### Software
```bash
conda create -n sentry python=3.11
conda activate sentry
pip install -r requirements.txt
pip install -e .
pip install scikit-learn
```

### Local Models (via Ollama)
Install Ollama from https://ollama.com/download, then pull the four models:

```bash
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull llama3.1:8b
ollama pull gemma2:9b
```

## Reproduce the Experiment

### Step 1: Start Ollama
```bash
ollama serve
```

### Step 2: Run the experiment
```bash
conda activate sentry
python -m sentry_jury.cli --config configs/mixed_experiment.yaml
```

### Step 3: View results
Results are saved to `runs/mixed_experiment_150/summary.json`.

## Expected Results

| Method | Clean Acc | Coverage | ASR |
|--------|-----------|----------|-----|
| single_best | 65.3% | 100% | 0% |
| majority_vote | 67.3% | 100% | low |
| clean_weighted | 65.3% | 100% | 0% |
| sentry | 65.3% | 100% | 0% |
| **meta_sentry** | **67.9%** | **93.3%** | **0%** |

## Key Files Changed

| File | Change |
|------|--------|
| `src/sentry_jury/meta_aggregator.py` | New — learned meta-aggregator |
| `src/sentry_jury/runner.py` | Added `meta_sentry` method |
| `src/sentry_jury/judges/ollama_judge.py` | New — Ollama local model judge |
| `src/sentry_jury/judges/factory.py` | Registered `ollama` backend |
| `src/sentry_jury/prompts.py` | Added 4 new prompt sensors |
| `configs/mixed_experiment.yaml` | 4 local models + meta_sentry config |

## Config Overview

```yaml
judges:
  - qwen2.5:7b
  - mistral:7b
  - llama3.1:8b
  - gemma2:9b

prompts:
  - safety_direct
  - safety_rubric

methods:
  - single_best
  - majority_vote
  - clean_weighted
  - sentry
  - meta_sentry

meta_sentry_coverage_floor: 0.80
```

## Notes

- The experiment uses disk-backed prediction cache. If you want to force a full re-run, delete the cache first: `Remove-Item -Recurse -Force .\runs\mixed_experiment_150`
- Models that fail to return valid JSON default to `safe (-1)` — this is intentional and conservative
- Total runtime: several hours depending on hardware
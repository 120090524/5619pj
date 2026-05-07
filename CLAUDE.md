CLAUDE.md
Project: SENTRY-Jury (CSE 5619 Security Course Project)
This is a group course project on robust LLM-as-a-Judge safety evaluation. I am one of 5 team members. The repo is shared. My role is the prompt dimension, aggregation layer analysis, and experiment running.

## What You Should Do First
Before making any changes, read:
- README.md
- docs/lab_log,md — full experiment history with decisions and findings (PRIMARY reference)
- runs/otto_prompt_ablation_v2_main/summary.json and profiles.json — latest results
- src/sentry_jury/ — package structure

Do not assume. Actually open and read.

## Project Goal
Prove that how you combine multiple LLM judges matters more than how many you use, and that SENTRY's attack-aware + dependence-aware + selective-abstention aggregation outperforms majority vote and clean_weighted baselines under adversarial attacks.

## Core Concept
Panelist = (judge_model, prompt) pair. Two panelists sharing the same model but different prompts are distinct — require separate inference calls. Panelist count = n_models × n_prompts.

## Current Status (as of 2026-04-21) — EXPERIMENTS COMPLETE

### My experiments (all done):
- **V1** (gpt-4o-mini × 5 prompts, 50/50): SENTRY = clean_weighted = majority_vote. Same-model constraint collapses dependence structure.
- **V2** (gpt-4o-mini × 5 prompts, 150 calib / 75 test): SENTRY = clean_weighted = majority_vote. Three methods fully identical, coverage 1.0.
- **V2c** (DeepSeek-R1-Distill-Qwen-7B × 5 prompts, 50/50, local inference): SENTRY ≈ clean_weighted. Same conclusion on a weaker model. Ran 10 hours on RTX 4090 with 4-bit NF4 quant.

### Teammate results (panwangying branch):
- 4-sensor (gpt-4o-mini + qwen_local × 2 prompts): SENTRY = clean_weighted, both slightly better than single_best on clean acc (0.878 vs 0.867). Abstention triggers (coverage 0.98).
- 6-sensor (+ mistral_local): All ensemble methods collapse to clean acc 0.687, FNR 0.827. single_best (0.873) is the clear winner. Mistral is too lenient, 4 local sensors outvote 2 API sensors.
- panwang also fixed two bugs: aggregator.py dependence penalty (was uniform, now relative to mean), runner.py probe family vote overwriting.

### Other teammate (qwen2.5:7b × 6 prompts, ToxicChat 300):
- SENTRY = clean_weighted. clean acc only 0.627 (model too weak). Same-model constraint same conclusion.

### Central finding across ALL experiments:
SENTRY has never outperformed clean_weighted in any configuration tested. Three compounding reasons when it fails:
1. Same model → dependence band too narrow, weights don't differentiate
2. Strong model → near-zero ASR → probe signal empty → nothing for SENTRY to respond to
3. Weak models outnumber strong → weight mechanism can't compensate for numerical disadvantage

### safety_strict finding (my contribution):
safety_strict has dependence=0.116 (genuinely heterogeneous) but accuracy=0.413 (systematically miscalibrated). The "when in doubt → unsafe" directive conflicts with JailJudge's labeling philosophy. A panelist can be heterogeneous for the wrong reason — SENTRY penalizes it but doesn't eliminate it, dragging ensemble below single_best on some attacks. Shows that independence alone is insufficient; panelists need to be both independent AND accurate.

### safety_strict interpretation:
Low accuracy is not a design failure — it reflects a mismatch between the prompt's judgment philosophy and the dataset's annotation standard. On a more conservatively-labeled dataset, safety_strict might be the best sensor.

## My 5 Prompt Variants (implemented in prompts.py)
- safety_direct: direct binary judgment, no reasoning space
- safety_rubric: detailed rubric, item-by-item evaluation
- safety_cot: chain-of-thought, reason first then decide
- safety_strict: default toward unsafe when ambiguous
- safety_lenient: default toward safe, intent-based holistic judgment

safety_strict and safety_lenient designed with opposite tiebreakers to maximize dependence spread.

## Code Changes I Made
- `src/sentry_jury/judges/hf_local_judge.py` — NEW: local HF model backend, 4-bit NF4 quantization, JSON parse fallback to keyword counting
- `runner.py` — disk cache persistence (_flush_cache/_load_cache), skip-calibration if profiles.json exists
- `litellm_judge.py` — think-block stripping, curly-quote normalization, 3-attempt retry

NOTE: panwangying branch has aggregator.py and runner.py fixes. If running fresh experiments, consider merging those fixes first.

## Environments
- `sentry` env: API-based runs (gpt-4o-mini via litellm)
- `cv-gpu` env: local HF model runs (has torch + bitsandbytes). Run with: `conda activate cv-gpu`, `set HF_HOME=D:/hf_cache`, `set TRANSFORMERS_OFFLINE=1`

## Run Commands
- API experiments: `conda activate sentry && python -B -m sentry_jury.cli --config configs/otto_prompt_ablation_v2_main.yaml`
- Local model: `conda activate cv-gpu && python -B -m sentry_jury.cli --config configs/otto_prompt_ablation_v2c_together.yaml`

## API Budget
- OpenAI Tier 1: 10,000 RPD. Test phase costs ~25 calls per example (5 sensors × (1 base + 4 probes)). 75 test examples ≈ 9,375 calls — fits. 100+ test examples will crash.
- No Tier 2 available.

## What's Next
- Writing report. Lab log (docs/lab_log,md) has full experiment history, decisions, and findings. Use it as the primary source.
- Report emphasis: effort, iteration, documented thinking. Results are consistently negative (SENTRY no-ops) but the analysis of WHY is the contribution.
- The safety_strict finding and the "conditions required for SENTRY to work" analysis are the two most reportable findings.

## Writing Style
- Direct, concise, no filler
- No em dashes
- No AI-formulaic phrasing ("it's worth noting", parallel triplets, "not just X but Y")
- Vary sentence length naturally

## Workflow
- Branch: otto (current)
- Shared repo via main on GitHub
- Work in VS Code on Windows 11, RTX 4090
- Scripts via terminal, results as JSON/CSV in runs/
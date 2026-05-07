# Reproduction Guide — Otto's Experiments (Branch: otto)

This branch implements the **prompt dimension** of SENTRY-Jury: five prompt variants
applied to a single judge model, studying whether prompt design alone creates enough
sensor diversity for SENTRY's aggregation to outperform majority vote.

Three experiments were run. V2 is the main result; V1 is a smaller smoke test;
V2c replaces the API judge with a local model.

---

## Prerequisites

### Environment A — API runs (V1, V2)

```bash
conda activate sentry
pip install -e .
```

Requires an OpenAI API key (Tier 1, 10,000 RPD limit):

```bash
set OPENAI_API_KEY=<your key>   # Windows
# or
export OPENAI_API_KEY=<your key>  # Linux/macOS
```

**API cost warning:** V2 makes ~9,375 calls (5 sensors × 25 calls/example × 75 test
examples). This fits within the 10,000 RPD Tier 1 limit but leaves little headroom.
Do not increase `test_limit` beyond 75 without a Tier 2 account.

### Environment B — Local model run (V2c)

```bash
conda activate cv-gpu   # must have torch + bitsandbytes installed
pip install -e .
```

Requires a GPU with at least 16 GB VRAM. Tested on RTX 4090.
The model (DeepSeek-R1-Distill-Qwen-7B) must be downloaded to your HF cache first:

```bash
set HF_HOME=<path to your hf_cache>
set TRANSFORMERS_OFFLINE=1   # prevents runtime download attempts
```

---

## Experiment V1 — Smoke test (50 calib / 50 test, API)

A smaller run to validate the pipeline before committing full API budget.

```bash
conda activate sentry
python -B -m sentry_jury.cli --config configs/otto_prompt_ablation_v1_smoke.yaml
```

Output: `runs/otto_prompt_ablation_v1_smoke/`

---

## Experiment V2 — Main result (150 calib / 75 test, API)

The primary experiment. Five prompt variants × gpt-4o-mini = 5 sensors.

```bash
conda activate sentry
python -B -m sentry_jury.cli --config configs/otto_prompt_ablation_v2_main.yaml
```

Output: `runs/otto_prompt_ablation_v2_main/`

Key files produced:
- `summary.json` — per-method accuracy, coverage, FNR, ASR across all attacks
- `profiles.json` — per-sensor base weight, dependence score, attack risk

Expected runtime: 20–40 minutes depending on API latency.

---

## Experiment V2c — Local model (50 calib / 50 test, DeepSeek-R1-Distill-Qwen-7B)

Same five prompt variants but with a local 4-bit quantized model instead of the API.

```bash
conda activate cv-gpu
python -B -m sentry_jury.cli --config configs/otto_prompt_ablation_v2c_together.yaml
```

Output: `runs/otto_prompt_ablation_v2c_local/`

Expected runtime: ~10 hours on RTX 4090 with 4-bit NF4 quantization.

---

## Prompt variants (defined in `src/sentry_jury/prompts.py`)

| Sensor name | Design intent |
|---|---|
| `safety_direct` | Direct binary judgment, no reasoning space |
| `safety_rubric` | Item-by-item rubric evaluation |
| `safety_cot` | Chain-of-thought: reason first, then decide |
| `safety_strict` | Default toward unsafe when ambiguous |
| `safety_lenient` | Default toward safe, intent-based holistic judgment |

`safety_strict` and `safety_lenient` were designed with opposite tiebreakers to
maximize dependence spread across the sensor panel.

---

## Code added in this branch

| File | Change |
|---|---|
| `src/sentry_jury/prompts.py` | Five prompt variants above |
| `src/sentry_jury/judges/hf_local_judge.py` | Local HF model backend, 4-bit NF4 quantization, JSON parse fallback |
| `src/sentry_jury/runner.py` | Disk cache persistence, skip-calibration if profiles.json exists |
| `src/sentry_jury/judges/litellm_judge.py` | Think-block stripping, curly-quote normalization, 3-attempt retry |
| `configs/otto_prompt_ablation_v1_smoke.yaml` | V1 config |
| `configs/otto_prompt_ablation_v2_main.yaml` | V2 config (main result) |
| `configs/otto_prompt_ablation_v2c_together.yaml` | V2c config (local model) |

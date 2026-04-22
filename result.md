# SENTRY-Jury Experiment Results — Panwang Ying

This README summarizes all experiments I ran on the `panwangying` branch for the SENTRY-Jury course project. Each experiment tests different judge combinations and dataset sizes to understand how the aggregation methods behave under real conditions.

---

## Code Changes

All experiments use the fixed pipeline with the following improvements I made:

- **`aggregator.py`** — Fixed dependence penalty to use deviation from mean, so SENTRY actually differentiates sensors instead of penalizing them uniformly
- **`runner.py`** — Fixed probe family vote overwriting bug (now merges votes from same-family probes)
- **`metrics.py`** — Added `attack_induced_abstain_rate` metric to track attack-triggered abstentions
- **`litellm_judge.py`** — Added exponential backoff retry for API calls

---

## Experiment 1: Mock Baseline (12 samples)

**Setup:** 4 mock judges × 2 prompts = 8 sensors, mock dataset of 12 samples.

**Purpose:** Validate pipeline logic end-to-end without API cost.

**Key finding:** Sensor weights showed clear differentiation after the fix (range: 0.136 to 1.454, a 10x spread), confirming that SENTRY's dependency-aware weighting works. However, with only 12 samples, the weight differences did not change final voting direction, so all methods produced identical numerical results.

| Method | Clean Acc. | Unsafe FNR |
|--------|-----------|-----------|
| single_best | 0.667 | 0.500 |
| majority_vote | 0.667 | 0.500 |
| clean_weighted | 0.667 | 0.500 |
| sentry | 0.667 | 0.500 |

---

## Experiment 2: Real API — 4 Sensors (gpt-4o-mini + qwen_local)

**Setup:** gpt4o_mini + qwen_local, 2 prompts each = 4 sensors. JailJudge 150 calibration + 150 test.

**Purpose:** First real-data experiment with a strong API judge and a local backup.

### Clean Results

| Method | Accuracy | Coverage | Unsafe FNR |
|--------|----------|----------|-----------|
| single_best | 0.867 | 1.00 | 0.019 |
| majority_vote | 0.867 | 1.00 | 0.038 |
| clean_weighted | **0.878** | 0.98 | 0.039 |
| sentry | **0.878** | 0.98 | 0.039 |

### Attack Success Rate (ASR, lower is better)

| Attack | single_best | majority_vote | clean_weighted | sentry |
|--------|-------------|---------------|----------------|--------|
| universal_phrase | **0.038** | 0.062 | 0.054 | 0.054 |
| prompt_injection | **0.000** | 0.008 | 0.008 | 0.008 |
| style_artifact | **0.023** | 0.092 | 0.093 | 0.093 |
| master_key | **0.008** | 0.015 | 0.016 | 0.016 |

**Key findings:**
- `clean_weighted` and `sentry` have the highest clean accuracy (0.878) by abstaining on ~2% of uncertain cases
- `majority_vote` is clearly worse than `single_best` on every attack, consistent with the mid-term report's finding that majority vote can be harmful when judges share weaknesses
- SENTRY = clean_weighted because only 2 judges produce symmetric dependence scores (same bug pattern as mid-term — 4 sensors is still not enough)

---

## Experiment 3: Real API — 6 Sensors (+ mistral_local)

**Setup:** gpt4o_mini + qwen_local + mistral_local, 2 prompts each = 6 sensors. JailJudge 150 calibration + 150 test.

**Purpose:** Add a third local judge to test if SENTRY's advantage appears with more sensors.

### Clean Results

| Method | Accuracy | Coverage | Unsafe FNR |
|--------|----------|----------|-----------|
| single_best | **0.873** | 1.00 | **0.019** |
| majority_vote | 0.687 | 1.00 | **0.827** ⚠️ |
| clean_weighted | 0.687 | 1.00 | **0.827** ⚠️ |
| sentry | 0.687 | 1.00 | **0.827** ⚠️ |

### Attack Success Rate (ASR)

| Attack | single_best | majority_vote | clean_weighted | sentry |
|--------|-------------|---------------|----------------|--------|
| universal_phrase | **0.038** | 0.078 | 0.078 | 0.078 |
| prompt_injection | **0.000** | 0.087 | 0.087 | 0.087 |
| style_artifact | **0.023** | 0.068 | 0.068 | 0.068 |
| master_key | **0.008** | 0.029 | 0.029 | 0.029 |

**Key findings:**
- `single_best` is the clear winner across all metrics — gpt-4o-mini alone outperforms any aggregation method
- All aggregation methods (majority_vote, clean_weighted, sentry) collapsed to `unsafe_fnr = 0.827`, meaning **82.7% of unsafe responses were mislabeled as safe**
- Root cause: Mistral-7B performs poorly on safety evaluation and tends to label everything as safe. When combined with qwen, the two weak judges (4 sensors total) outvote the strong gpt-4o-mini (2 sensors)
- This is a striking real-world demonstration of the mid-term report's warning: **"more judges does not mean more robust"** — adding a weak judge can actively destroy the aggregation quality

---

## Comparison Across Experiments

| Experiment | Sensors | Clean Acc (best) | Best Method | Key Takeaway |
|-----------|---------|------------------|-------------|--------------|
| 1: Mock | 8 | 0.667 | all tied | Validates pipeline logic |
| 2: 4 sensors (real API) | 4 | 0.878 | clean_weighted = sentry | Weighted methods improve over majority_vote |
| 3: 6 sensors (real API) | 6 | 0.873 | single_best | Weak judges can destroy the jury |

---

## Main Conclusions

1. **Sensor quality matters more than sensor count.** Adding Mistral-7B as a judge reduced overall accuracy from 0.878 to 0.687 because it is systematically biased toward "safe." This confirms the mid-term report's core claim that naive scaling hurts performance.

2. **SENTRY still matches clean_weighted.** Even with 6 sensors, SENTRY produces identical numerical results to clean_weighted in these experiments. The reason is that when multiple weak judges share the same failure mode (both mistral sensors + both qwen sensors vote "safe"), dependence scores become nearly symmetric again, and the relative penalty cannot distinguish them. SENTRY's advantage requires judges with *diverse* failure modes, not just more judges.

3. **Single-best remains strong when one judge dominates.** When gpt-4o-mini is substantially better than the alternatives, combining it with weaker judges only dilutes its signal. This is a meaningful negative result for the aggregation approach.

4. **The `attack_induced_abstain_rate` metric I added captured new behavior.** In Experiment 2, clean_weighted and sentry showed small but nonzero abstain rates under style_artifact and master_key attacks (0.68%), indicating the defensive abstention logic is triggering correctly on hard examples.

---

## Files Changed on `panwangying` Branch

| File | Change |
|------|--------|
| `src/sentry_jury/aggregator.py` | Relative dependence penalty |
| `src/sentry_jury/runner.py` | Probe family vote merging |
| `src/sentry_jury/metrics.py` | New `attack_induced_abstain_rate` metric |
| `src/sentry_jury/judges/litellm_judge.py` | Exponential backoff retry |
| `configs/course_project_mock.yaml` | Added qwen_local and mistral_local judges, switched to JailJudge dataset |

---

## Next Steps

- **Replace Mistral with a stronger local model** (e.g., Llama 3.1 8B) to see if SENTRY's advantage emerges when all judges are at least moderately competent
- **Scale to 300+ test samples** for more stable ASR estimates, especially for rare attack-success events
- **Add a second strong API judge** (e.g., gpt-4o or Claude) to balance the jury so one model does not dominate
- **Run per-category analysis** on JailJudge to see if SENTRY helps more on specific unsafe content types
- 

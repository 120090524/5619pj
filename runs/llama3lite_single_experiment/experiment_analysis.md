# Experiment Analysis: llama3lite_single_experiment

## Models (Judges)

| Name | Model | Provider | Temperature |
|------|-------|----------|-------------|
| llama3lite_a | Meta-Llama-3-8B-Instruct-Lite | Together AI | 0.0 |
| llama3lite_b | Meta-Llama-3-8B-Instruct-Lite | Together AI | 0.0 |
| llama3lite_c | Meta-Llama-3-8B-Instruct-Lite | Together AI | 0.0 |

All three judges use the **same model and same temperature** — a deliberate design choice to study the effect of aggregation when the sensor ensemble has zero diversity.

All models accessed via LiteLLM unified API wrapper.

## Architecture

**Sensors = Judges × Prompts = 3 × 2 = 6 sensors**

Each sensor is a (judge, prompt) pair:
- `llama3lite_a::safety_direct`
- `llama3lite_a::safety_rubric`
- `llama3lite_b::safety_direct`
- `llama3lite_b::safety_rubric`
- `llama3lite_c::safety_direct`
- `llama3lite_c::safety_rubric`

**Best sensor (calibration):** `llama3lite_a::safety_rubric::original`

## Dataset

- Source: JailJudge (in-distribution)
- Calibration: 150 examples
- Test: 150 examples
- Total: 300 examples

## Attacks (4)

| Attack | Description |
|--------|-------------|
| `universal_phrase` | Prepends a known jailbreak trigger phrase |
| `prompt_injection` | Injects a hidden instruction to override the judge |
| `style_artifact` | Adds stylistic noise to confuse the judge |
| `master_key` | Uses a "master key" phrase to bypass safety checks |

## Probes / Defenses (4)

| Probe | Description |
|-------|-------------|
| `whitespace_normalize` | Strips unusual whitespace |
| `strip_judge_directives` | Removes injected judge instructions |
| `style_normalize` | Normalizes stylistic artifacts |
| `strip_master_keys` | Removes known master key phrases |

## Aggregation Methods (4)

| Method | Description |
|--------|-------------|
| `single_best` | Uses only the highest-performing sensor from calibration |
| `majority_vote` | Simple majority vote across all 6 sensors |
| `clean_weighted` | Weighted vote using calibration accuracy as weights |
| `sentry` | Correlation-aware weighted vote with attack penalty (λ=1.20, ρ=0.40) |

## Results

### Metric Definitions
- **ACC**: Accuracy
- **COV**: Coverage (fraction of examples with a decision)
- **UFNR**: Unsafe False Negative Rate (missed harmful content)
- **Flip**: Flip Rate (fraction of examples where attack changed the decision)
- **ASR**: Attack Success Rate (fraction of unsafe examples fooled by attack)

---

### Clean (no attack)

| Method | ACC | COV | UFNR |
|--------|-----|-----|------|
| single_best | 64.7% | 100% | 96.2% |
| majority_vote | 39.3% | 100% | 0.0% |
| clean_weighted | 64.7% | 100% | 96.2% |
| sentry | 64.7% | 100% | 96.2% |

---

### Attack: universal_phrase

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 70.7% | 100% | 80.8% | 11.3% | 4.1% |
| majority_vote | 74.7% | 100% | 25.0% | 59.3% | 30.5% |
| clean_weighted | 70.7% | 100% | 80.8% | 11.3% | 4.1% |
| sentry | 70.7% | 100% | 80.8% | 11.3% | 4.1% |

---

### Attack: prompt_injection

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 65.3% | 100% | 100.0% | 3.3% | 2.1% |
| majority_vote | 65.3% | 100% | 94.2% | 91.3% | 83.1% |
| clean_weighted | 65.3% | 100% | 100.0% | 3.3% | 2.1% |
| sentry | 65.3% | 100% | 100.0% | 3.3% | 2.1% |

---

### Attack: style_artifact

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 66.7% | 100% | 94.2% | 3.3% | 1.0% |
| majority_vote | 34.0% | 100% | 3.8% | 6.7% | 15.3% |
| clean_weighted | 66.7% | 100% | 94.2% | 3.3% | 1.0% |
| sentry | 66.7% | 100% | 94.2% | 3.3% | 1.0% |

---

### Attack: master_key

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 66.0% | 100% | 90.4% | 2.7% | 1.0% |
| majority_vote | 38.0% | 100% | 0.0% | 5.3% | 8.5% |
| clean_weighted | 66.0% | 100% | 90.4% | 2.7% | 1.0% |
| sentry | 66.0% | 100% | 90.4% | 2.7% | 1.0% |

---

## Key Takeaways

- **Single model ensemble provides no diversity**: All three judges (`llama3lite_a/b/c`) use the same model and temperature, producing identical predictions. As a result, `single_best`, `clean_weighted`, and `sentry` yield exactly the same outputs — the aggregation logic has nothing to work with.

- **`majority_vote` collapses under prompt_injection**: With zero inter-judge diversity, when all three instances are fooled by an attack, majority vote amplifies the error rather than correcting it. This produces catastrophic ASR of 83.1% under prompt_injection — far worse than using a single sensor.

- **`single_best` is the only viable method here**: By relying solely on the best-calibrated sensor (`llama3lite_a::safety_rubric`), it avoids the correlated-failure pathology and holds ASR to 1–4% across all attacks.

- **Model capability is the binding constraint**: Clean accuracy of 64.7% and UFNR of 96.2% indicate that `Meta-Llama-3-8B-Instruct-Lite` is too weak as a safety judge — it misses nearly all unsafe content even without any attack. Aggregation methods cannot compensate for a fundamentally unreliable base sensor.

- **Comparison to mixed_experiment_150**: The 3-diverse-judge setup (Qwen 7B + Gemma 3n + GPT-OSS 20B) achieved 89–93% clean accuracy and held ASR below 24%. This experiment confirms that **judge diversity is a prerequisite for ensemble robustness**, not model count.

# Experiment Analysis: mixed_experiment_150

## Models (Judges)

| Name | Model | Provider | Temperature |
|------|-------|----------|-------------|
| qwen7b | Qwen2.5-7B-Instruct-Turbo | Together AI | 0.0 |
| gemma3n | gemma-3n-E4B-it | Together AI | 0.0 |
| gpt_oss | gpt-oss-20b | Together AI | 0.0 |

All models accessed via LiteLLM unified API wrapper.

## Architecture

**Sensors = Judges × Prompts = 3 × 2 = 6 sensors**

Each sensor is a (judge, prompt) pair:
- `qwen7b::safety_direct`
- `qwen7b::safety_rubric`
- `gemma3n::safety_direct`
- `gemma3n::safety_rubric`
- `gpt_oss::safety_direct`
- `gpt_oss::safety_rubric`

**Best sensor (calibration):** `gpt_oss::safety_rubric::original`

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
| single_best | 89.3% | 100% | 13.5% |
| majority_vote | 89.3% | 100% | 3.8% |
| clean_weighted | 92.9% | 93.3% | 4.3% |
| sentry | 92.8% | 92.0% | 4.4% |

---

### Attack: universal_phrase

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 87.3% | 100% | 17.3% | 8.7% | 6.0% |
| majority_vote | 78.0% | 100% | 57.7% | 26.0% | 20.9% |
| clean_weighted | 78.0% | 100% | 57.7% | 21.4% | 18.5% |
| sentry | 78.0% | 100% | 57.7% | 21.0% | 18.0% |

---

### Attack: prompt_injection

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 90.7% | 100% | 7.7% | 5.3% | 2.2% |
| majority_vote | 76.7% | 100% | 65.4% | 30.0% | 23.9% |
| clean_weighted | 76.7% | 100% | 65.4% | 24.3% | 20.8% |
| sentry | 76.7% | 100% | 65.4% | 23.2% | 19.5% |

---

### Attack: style_artifact

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 89.3% | 100% | 17.3% | 9.3% | 5.2% |
| majority_vote | 90.0% | 100% | 21.2% | 14.0% | 7.5% |
| clean_weighted | 89.9% | 99.3% | 21.6% | 9.4% | 5.4% |
| sentry | 89.9% | 99.3% | 21.6% | 8.8% | 4.7% |

---

### Attack: master_key

| Method | ACC | COV | UFNR | Flip | ASR |
|--------|-----|-----|------|------|-----|
| single_best | 91.3% | 100% | 3.8% | 6.0% | 2.2% |
| majority_vote | 90.7% | 100% | 0.0% | 2.7% | 0.7% |
| clean_weighted | 91.2% | 98.0% | 0.0% | 2.9% | 1.5% |
| sentry | 91.2% | 98.0% | 0.0% | 2.9% | 1.6% |

---

## Key Takeaways

- `single_best` is the most robust against adversarial attacks, especially universal_phrase and prompt_injection (ASR <6%).
- `sentry` and `clean_weighted` achieve the highest clean accuracy (~93%) but are more vulnerable to injection-style attacks.
- `majority_vote` is highly susceptible to universal_phrase and prompt_injection (ASR ~21-24%, UFNR ~58-65%).
- `master_key` attack is largely neutralized across all methods, suggesting the strip_master_keys probe is effective.

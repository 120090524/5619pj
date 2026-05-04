# Reuse notes for published codebases

This file is intentionally practical: **what should you borrow first, and what should you avoid copying blindly?**

## Best starting points

### 1) JudgeBench
Why it is useful:
- clean JSONL pairwise format,
- a simple `Judge` abstraction,
- a `run_judge.py` style entry point,
- examples of prompt-based judges and reward models.

What to borrow:
- input schema,
- single-judge runner shape,
- prompt-template organization,
- async evaluation pattern.

Caution:
- the GitHub page clearly documents the repo and the JSONL format, but I was **not able to confidently verify the code repository license from the GitHub page** during preparation. Check the repository before copying code directly.

### 2) JAILJUDGE
Why it is useful:
- safety/jailbreak benchmark organization,
- multi-agent judge framing,
- broad-risk and multilingual test splits.

Good news:
- the GitHub page shows **MIT license**.

What to borrow:
- dataset organization,
- benchmark naming and split structure,
- safety-evaluation framing.

### 3) Prometheus-Eval
Why it is useful:
- practical evaluator API,
- local inference through vLLM,
- API inference through LiteLLM,
- strong open evaluator baseline.

Good news:
- the GitHub page shows **Apache-2.0 license**.

What to borrow:
- judge wrapper ideas,
- API configuration pattern,
- prompt/response parsing style.

### 4) JudgeLM
Why it is useful:
- open-source judge models,
- evaluation and serving structure,
- reference/swap augmentation ideas.

Good news:
- the GitHub page shows **Apache-2.0 license**.

What to borrow:
- open judge backends,
- serving/evaluation structure,
- prompt layout for pairwise judging.

### 5) Judge Reliability Harness (JRH)
Why it is useful:
- probe families,
- reliability stress tests,
- formatting/paraphrase/verbosity/stochasticity checks.

Good news:
- the RAND GitHub organization page shows the repo is **MIT licensed**.

What to borrow:
- reliability test taxonomy,
- pipeline staging for perturbation-based evaluation.

## Useful but license-sensitive

### RobustJudge
Why it is useful:
- attack taxonomy,
- defense taxonomy,
- modular robustness workflow,
- prompt-template sensitivity analysis.

Important caution:
- the GitHub page shows **GPL-3.0 license**.

This means:
- reading the code and re-implementing the ideas is fine,
- but **copy-pasting** code into a repo you want to keep MIT/Apache is risky,
- if you directly include GPL-3.0 code, the resulting combined project may need GPL-3.0 compliance.

Recommended use:
- borrow the experiment design,
- re-implement the minimal pieces you need.

## Safe implementation strategy for this course project

1. Use **your own repo and your own aggregation code**.
2. Use:
   - JudgeBench/JAILJUDGE for data format inspiration,
   - Prometheus-Eval/JudgeLM as judge backends,
   - JRH for probe ideas.
3. Read RobustJudge for attack organization, but do not paste code unless you are okay with GPL-3.0.

## A good concrete plan

### Option A: fastest
- Keep this scaffold.
- Replace mock judges with LiteLLM/OpenAI/Prometheus judges.
- Load your benchmark into the internal JSONL format.
- Run the same pipeline.

### Option B: hybrid with JudgeBench
- Start from JudgeBench runner layout.
- Add the selective aggregation and attack/probe modules from this scaffold.
- Keep your new aggregation code in a separate folder.

### Option C: safety-first
- Keep this scaffold.
- Use JudgeZoo or JAILJUDGE-compatible safety judges.
- Focus the whole project on safety classification only.
- Mention JudgeBench only as transfer/generalization in the proposal.

## Internal schema used in this scaffold

### Safety binary
```json
{
  "example_id": "s1",
  "task_type": "safety_binary",
  "prompt": "user request",
  "response_a": "model response",
  "label": 1
}
```

Label:
- `1` = unsafe / policy-violating response
- `-1` = safe / refusal / benign response

### Pairwise
```json
{
  "example_id": "p1",
  "task_type": "pairwise",
  "prompt": "question",
  "response_a": "candidate A",
  "response_b": "candidate B",
  "label": 1
}
```

Label:
- `1` = A should win
- `-1` = B should win

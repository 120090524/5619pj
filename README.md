# SENTRY-Jury Course Project Scaffold

A course-project-ready scaffold for studying **robust LLM safety evaluation** with an **attack-aware, dependence-aware, selective jury**.

This repo is designed to be **small enough to finish for a class project**, while still matching the proposal:
- frozen judges (no judge fine-tuning),
- multiple judge x prompt sensors,
- attack calibration,
- probe-based instance risk,
- selective aggregation with abstention.

## What is included

1. `proposal/` — a LaTeX proposal you can submit or edit.
2. `src/sentry_jury/` — a minimal Python package with:
   - judge interfaces,
   - a deterministic mock judge for local demos,
   - optional LiteLLM API judge wrapper,
   - attack generators,
   - probe generators,
   - profile estimation,
   - selective aggregation,
   - metrics and experiment runner.
3. `configs/` — a small mock config and an API-template config.
4. `data/` — a tiny synthetic safety dataset for end-to-end smoke tests.
5. `docs/reuse_notes.md` — notes on which published repos are good starting points and which licenses to watch.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m sentry_jury.cli --config configs/course_project_mock.yaml
```

Outputs will be written under `runs/mock_course_project/`.

## What the demo does

The demo uses three **mock judges** with different attack sensitivities. It calibrates on half of the toy safety dataset, then evaluates:
- clean performance,
- attacked performance for four attack families,
- abstention / coverage,
- attack success rate and flip rate.

This lets you verify the pipeline logic before spending API budget.

## How to switch to real judges

Edit `configs/api_template.yaml`:

```yaml
judges:
  - name: gpt4o_judge
    backend: litellm
    model: openai/gpt-4o-mini
  - name: prometheus2
    backend: litellm
    model: huggingface/prometheus-eval/prometheus-7b-v2.0
```

Then run:

```bash
export OPENAI_API_KEY=...
python -m sentry_jury.cli --config configs/api_template.yaml
```

## Recommended borrowing strategy

For a course project, the easiest path is:

1. **Borrow data format and single-judge runner ideas from JudgeBench.**
2. **Borrow safety benchmark organization from JAILJUDGE.**
3. **Use Prometheus-Eval / JudgeLM / JudgeZoo as your judge backends.**
4. **Borrow attack categories and defense naming from RobustJudge.**
5. **Borrow reliability test ideas from Judge Reliability Harness (JRH).**
6. Keep your own aggregation code lightweight and original.

More detail is in `docs/reuse_notes.md`.

## Suggested repo structure

```text
sentry-jury-course-project/
├── proposal/
│   ├── proposal.tex
│   └── references.bib
├── configs/
│   ├── course_project_mock.yaml
│   └── api_template.yaml
├── data/
│   └── sample_safety_eval.jsonl
├── docs/
│   └── reuse_notes.md
├── scripts/
│   ├── init_git_repo.sh
│   └── run_mock_demo.sh
├── src/
│   └── sentry_jury/
│       ├── __init__.py
│       ├── aggregator.py
│       ├── attacks.py
│       ├── cli.py
│       ├── datasets.py
│       ├── metrics.py
│       ├── profiles.py
│       ├── prompts.py
│       ├── probes.py
│       ├── runner.py
│       ├── types.py
│       ├── utils.py
│       └── judges/
│           ├── __init__.py
│           ├── base.py
│           ├── factory.py
│           ├── litellm_judge.py
│           └── mock.py
├── .gitignore
├── LICENSE
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Turning this into a GitHub repo

```bash
git init
git branch -m main
git add .
git commit -m "Initial course project scaffold for SENTRY-Jury"

git remote add origin <your-github-repo-url>
git push -u origin main
```

A helper script is also provided:

```bash
bash scripts/init_git_repo.sh
```

## Scope recommendation for a class project

Keep the first full experiment small:
- 1 main safety benchmark,
- 1 optional transfer benchmark,
- 3 judges,
- 2 prompts,
- 4 attack families,
- 4-5 baselines.

That is already enough for a strong course project.

## License

The scaffold in this repository is released under the MIT License.

Important: if you **copy code** from GPL-3.0 projects such as RobustJudge, your combined repo may need to comply with GPL-3.0. Read `docs/reuse_notes.md` before copying upstream code.

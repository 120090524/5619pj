CLAUDE.md
Project: SENTRY-Jury (CSE 5619 Security Course Project)
This is a group course project on robust LLM-as-a-Judge safety evaluation. I am one of 5 team members. The repo is shared. My role for the remainder of the project is primarily the prompt dimension, aggregation layer analysis, and experiment running — not the model inference infrastructure (teammates are handling that).
What You Should Do First
Before making any changes, read these to understand the current state:

README.md — repo overview and how to run
proposal/proposal.tex (or rendered version) — the method specification
configs/course_project_mock.yaml and configs/api_template.yaml — to understand the panelist config schema
src/sentry_jury/ — walk the package structure. Pay attention to:

runner.py — the experiment loop
aggregator.py — where SENTRY vs majority vote vs clean_weighted is implemented
profiles.py — calibration and dependence estimation
probes.py — probe perturbations for online risk
attacks.py — attack generators
judges/ — judge backends (mock, litellm, and likely local HF models)
prompts.py — this file is critical for my work. Check if prompt variants are defined here or in configs.


runs/ — if teammates have pushed experiment outputs, read the latest to see what panelist configurations have already been tried

Do not assume. Actually open and read. The mid-term report (described below) may be out of date relative to main.
Project Goal (One Sentence)
Prove that how you combine multiple LLM judges matters more than how many you use, and that SENTRY's attack-aware + dependence-aware + selective-abstention aggregation outperforms majority vote and clean_weighted baselines — especially under adversarial attacks on the judges.
Core Concept You Must Internalize
Panelist = (judge_model, prompt) pair. This is the project's central definition, straight from the proposal:

"Even if two panelists share the same underlying model, they are treated as distinct if their instructions differ."

Implication: two panelists sharing the same model but using different prompts require two independent inference calls per sample, not one call that outputs two judgments. Don't collapse them.
This means panelist count = n_models × n_prompts (assuming full cross). Teammates have been scaling the model dimension (Llama-8b, Qwen-7b, Qwen-14b, Gemma, gpt-4o-mini). The prompt dimension is likely underdeveloped and is my contribution area.
Current Status (as of mid-term)

Pipeline is end-to-end functional: panelist profiling, calibration, probe-based risk, selective aggregation with abstention
Two benchmarks run: JailbreakBench JBB-Behaviors subset (100 samples, 2 attacks) and JailJudge general (300 samples, 4 attacks)
Four aggregation methods compared: single_best, majority_vote, clean_weighted, sentry
Four attack types supported: universal safe-looking phrases, prompt injection, style artifacts, master-key prefix

Key Finding from Mid-Term
SENTRY and clean_weighted produce nearly identical results. Reason: mid-term only used 2 panelists. With N=2 there's no meaningful dependence structure to model and probe signals have nothing to differentiate. SENTRY's extra mechanisms essentially no-op and it degrades to clean_weighted.
Separately confirmed: majority_vote is not a safe default. In some attack settings (e.g. Benchmark I prompt injection ASR=0.0625, Benchmark II master-key flip rate), it is worse than single_best. This finding is already solid and should be preserved.
Recent Progress (Post Mid-Term, from teammates)
Teammates are now running 6 panelists. A preview result from a teammate shows dependence values in [0.17, 0.69] range — better spread than N=2 but still room to grow by adding heterogeneous judges. I need to verify from the repo exactly what the current 6-panelist configuration looks like (is it 6 models × 1 prompt, 3 models × 2 prompts, etc.) before I add anything.
Teammates are running local 7B/8B/14B models (Qwen-7b, Llama-8b, Gemma, Qwen-14b). Each teammate typically runs a different model on their own GPU. This is compute distribution, not method design — they are splitting inference load, not expanding the panelist design space.
My Role and Immediate Next Steps
Priority 1: Understand current panelist configuration
Read the configs and any recent runs to answer:

Exactly which (model, prompt) pairs are currently active?
Is there a prompt variant library, or does every panelist use the same underlying prompt template?
Where are prompts defined — in prompts.py, in YAML configs, or hardcoded in the judge wrapper?

Priority 2: Build a prompt variant library
If prompt diversity is missing or thin, design at least 3–5 meaningfully distinct safety evaluation prompts. Not paraphrases — different evaluation strategies. Planned directions:

safety_rubric_strict: detailed rubric, any violation → unsafe
safety_rubric_lenient: holistic intent-based judgment
chain_of_thought: reason first, then decide
direct_binary: no reasoning space, force safe/unsafe output
role_expert: role-play as safety auditor

Each prompt should be implementable such that pairing it with any existing judge model creates a valid, distinct panelist.
Priority 3: Run experiments and analyze
After prompt variants are in, re-run the benchmarks with expanded panelist configurations (target: model × prompt cross, realistic size 6–12 panelists). Compare SENTRY vs baselines. The key question: does SENTRY meaningfully diverge from clean_weighted once panelist diversity is increased?
Also intentionally include same-family model pairs (e.g. Qwen-7b + Qwen-14b) to stress-test the dependence-aware redundancy penalty — this is a direct experimental lever for showing the method's value.
What Professor Cares About
The grading emphasis is on effort, iteration, and documented thinking — not just final numeric results. This affects how we work:

Keep failed experiments. Don't delete bad run outputs; archive them under runs/ with clear names and dates.
Git commit messages should explain what and why, not just "update" or "fix".
Maintain a lab log (I'll keep one in a local file, possibly docs/lab_log.md later) documenting decisions and dead ends.
If a code change is experimental or speculative, comment it as such.

Workflow Constraints

I work in VS Code on Windows 11, RTX 4090, with a conda env dedicated to this project (separate from my CV course env)
I do not run large local models myself — teammates handle that on their machines. I run the aggregation/calibration layer and API-based judges (gpt-4o-mini) locally
I run scripts via terminal, not Jupyter. I inspect results as JSON/CSV files in runs/
The repo is shared via main on GitHub. I work on feature branches (e.g. otto/prompt-variants) and submit via PR

Writing Style for Any Code Comments, Docs, or Report Contributions

Direct, concise, no filler
No em dashes
No AI-formulaic phrasing (no "it's worth noting", no parallel triplets, no "not just X but Y" cadence)
Vary sentence length naturally
If pointing out a problem in existing code or a teammate's approach, state it directly with a reason

Things to Verify in the Repo Before Making Changes
Please actually check these rather than assuming:

Does the current config schema support (model, prompt) cross-product explicitly, or does it treat each panelist entry as a standalone unit?
Where exactly are prompts stored? Do configs/*.yaml reference prompt names that resolve to template files, or are prompt strings inlined?
Does profiles.py actually compute dependence/correlation between panelists, and is that value used by the sentry aggregator? Trace the data flow.
How does aggregator.py decide to abstain? What's the evidence threshold?
What has already been committed post mid-term? Check recent commits on main and any branches. Teammates may have already added things I don't know about.

Report what you find before suggesting changes. I'd rather have an accurate picture than a fast wrong one.

When I give you a task, start by acknowledging what you already know from this file, then ask any clarifying questions before reading/editing files. If I ask for a change and the repo's actual state contradicts something in this doc, trust the repo, flag the discrepancy, and tell me.
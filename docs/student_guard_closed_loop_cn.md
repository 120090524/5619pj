# 基于 `boyu` 分支的 Student Guard 训练闭环教程（新手版）

这份教程的目标很简单：

1. 先把你本地仓库从 `main` 更新到 `boyu` 分支。
2. 再把我给你的补丁代码放进仓库。
3. 用现有的 SENTRY teacher 去挖你本地小模型被攻破的失败样本。
4. 把这些失败样本整理成 **student guard** 训练集。
5. 训练一个本地小型安全分类器（compact student guard）。
6. 把训练好的 student guard 接回原来的 `sentry_jury` 框架继续评测。

---

## 1. 先说清楚：你现在训练的到底是什么？

你现在这个仓库最适合训练的不是“新的回复模型”，而是一个：

> **output-level student guard**

它的输入是：

- 用户请求 `prompt`
- 本地模型真实生成的回复 `response`

它的输出是：

- `safe`
- `unsafe`

所以它本质上是一个 **安全分类器 / 本地 guard / 本地 student judge**。

这条线最适合你当前 repo，因为你当前 repo 的核心 schema 本来就是：

- `prompt`
- `response_a`
- `label`

而且当前 `safety_direct` / `safety_rubric` prompt 也是在判断 **用户请求 + 模型回复** 是否 unsafe。

---

## 2. 这个闭环的完整流程图

你最终会跑这样一条线：

```text
原始 safety prompts
    -> prompt-side target attacks
    -> 本地 target model 生成 response
    -> SENTRY teacher 判断 response 是否 unsafe
    -> 保存 hard negatives
    -> 构造 classification / preference 数据
    -> 训练 compact student guard
    -> student guard 回接 sentry_jury 继续评测
```

你现在拿到的补丁代码就是围绕这条线写的。

---

## 3. 先把本地 `main` 切换到 `boyu`

### 3.1 先进入你的项目目录

```bash
cd /你的路径/5619pj
```

### 3.2 看看你当前是不是在 `main`

```bash
git branch --show-current
```

如果输出是：

```bash
main
```

说明你现在就在本地 `main`。

### 3.3 先看你有没有没保存的修改

```bash
git status
```

如果你看到很多 `modified:` 或 `untracked files:`，不要直接切分支。

你有两种安全做法。

#### 做法 A：直接提交到你当前本地 `main`

如果这些修改你想保留：

```bash
git add .
git commit -m "save local main work before switching to boyu"
```

#### 做法 B：先临时收起来（更适合新手）

如果你只是想暂存：

```bash
git stash push -u -m "temp stash before switching to boyu"
```

> `-u` 的意思是把未跟踪文件也一起 stash。

### 3.4 从远端更新分支信息

```bash
git fetch origin
```

### 3.5 看看远端有没有 `boyu`

```bash
git branch -r
```

你应该能看到类似：

```bash
origin/main
origin/boyu
```

### 3.6 第一次在本地创建并切到 `boyu`

如果你本地还没有 `boyu`，执行：

```bash
git switch -c boyu origin/boyu
```

如果你的 Git 版本比较老，也可以用：

```bash
git checkout -b boyu origin/boyu
```

### 3.7 确认你真的切过去了

```bash
git branch --show-current
```

应该输出：

```bash
boyu
```

### 3.8 以后再更新 `boyu`

以后你已经有本地 `boyu` 了，就这样更新：

```bash
git switch boyu
git pull
```

---

## 4. 把补丁代码复制到你的 repo 里

你拿到的补丁目录里有这些文件：

```text
configs/student_guard_eval.yaml

docs/student_guard_closed_loop_cn.md

scripts/mine_local_failures.py
scripts/build_guard_datasets.py
scripts/evaluate_student_guard.py

src/sentry_jury/teacher.py
src/sentry_jury/target_attacks.py
src/sentry_jury/judges/hf_classifier.py
src/sentry_jury/judges/factory.py        <- 替换原文件
src/sentry_jury/judges/__init__.py       <- 替换原文件

training/requirements-guard.txt
training/train_guard_classifier.py
```

### 最简单复制方法

假设你下载的补丁解压目录叫：

```text
~/Downloads/5619pj_boyu_student_guard_patch
```

那你可以在 repo 根目录执行：

```bash
cp -r ~/Downloads/5619pj_boyu_student_guard_patch/configs/* configs/
cp -r ~/Downloads/5619pj_boyu_student_guard_patch/docs/* docs/
cp -r ~/Downloads/5619pj_boyu_student_guard_patch/scripts/* scripts/
cp -r ~/Downloads/5619pj_boyu_student_guard_patch/src/sentry_jury/* src/sentry_jury/
cp -r ~/Downloads/5619pj_boyu_student_guard_patch/src/sentry_jury/judges/* src/sentry_jury/judges/
mkdir -p training
cp -r ~/Downloads/5619pj_boyu_student_guard_patch/training/* training/
```

复制完以后，建议马上看一下：

```bash
git status
```

你应该能看到新增和修改的文件。

---

## 5. 创建环境并安装依赖

### 5.1 建议单独开一个虚拟环境

```bash
python -m venv .venv
```

Mac / Linux：

```bash
source .venv/bin/activate
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5.2 先安装原 repo 依赖

```bash
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 5.3 再安装训练依赖

```bash
pip install -r training/requirements-guard.txt
```

如果你没有 NVIDIA GPU，可以先不要用 `bitsandbytes`，即使安装失败也不影响 CPU / 普通训练的第一版。

---

## 6. 第一步：先用原仓库验证 teacher 能跑

你可以先跑原来的配置，确认 `boyu` 分支正常：

```bash
python -m sentry_jury.cli --config configs/api_jailjudge_id.yaml
```

如果你现在没有 API key，不想先花钱，那就至少跑 mock：

```bash
python -m sentry_jury.cli --config configs/course_project_mock.yaml
```

只要这里能跑，后面的 teacher 复用才会稳定。

---

## 7. 第二步：挖本地 target model 的失败样本

这一步会做：

1. 读入 safety prompt 数据。
2. 对 prompt 做 prompt-side attack。
3. 让本地 target model 生成回复。
4. 用现有 SENTRY teacher 判断这个回复是不是 unsafe。
5. 保存高置信 hard negatives。

### 7.1 你先准备一个本地 target model

最容易起步的选择：

- `Qwen/Qwen2.5-1.5B-Instruct`
- 或你自己本地已经下载好的小模型目录

如果你是 Hugging Face 模型名，`--model` 就写模型名。
如果你已经下载到本地，`--model` 就写本地路径。

### 7.2 运行挖失败样本脚本

下面是最推荐的新手命令：

```bash
python scripts/mine_local_failures.py \
  --teacher-config configs/api_jailjudge_id.yaml \
  --input-jsonl data/final/jailjudge_id_calib.jsonl data/final/jailjudge_id_clean_eval.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-jsonl data/student_guard/mined_failures.jsonl \
  --prompt-limit 200 \
  --attack-names identity,roleplay,prefix_injection,translation_bridge,obfuscation \
  --max-new-tokens 256 \
  --do-sample
```

如果你 GPU 显存紧张，可以加：

```bash
--load-in-4bit
```

完整示例：

```bash
python scripts/mine_local_failures.py \
  --teacher-config configs/api_jailjudge_id.yaml \
  --input-jsonl data/final/jailjudge_id_calib.jsonl data/final/jailjudge_id_clean_eval.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-jsonl data/student_guard/mined_failures.jsonl \
  --prompt-limit 200 \
  --attack-names identity,roleplay,prefix_injection,translation_bridge,obfuscation \
  --max-new-tokens 256 \
  --do-sample \
  --load-in-4bit
```

### 7.3 跑完后你会得到什么

你会看到：

```text
data/student_guard/mined_failures.jsonl
data/student_guard/mined_failures.summary.json
```

`mined_failures.jsonl` 每一行都包含：

- 原始 prompt
- attack 后 prompt
- 本地 target model 的回复
- teacher 的最终判断
- teacher confidence
- sensor weights
- instance risks

这是整个闭环最关键的中间文件。

---

## 8. 第三步：把 mined failures 变成训练集

这一步会同时产出三套数据：

1. `classification_*.jsonl`：给 student guard 分类训练用。
2. `preference_*.jsonl`：给你以后做 DPO / reward modeling 预留。
3. `eval_examples_*.jsonl`：继续接回原 repo 评测用。

### 8.1 运行命令

```bash
python scripts/build_guard_datasets.py \
  --mined-jsonl data/student_guard/mined_failures.jsonl \
  --extra-safe-jsonl data/final/jailjudge_id_calib.jsonl data/final/jailjudge_id_clean_eval.jsonl \
  --output-dir data/student_guard \
  --min-teacher-confidence 0.80
```

### 8.2 跑完以后会生成什么

```text
data/student_guard/classification_train.jsonl
data/student_guard/classification_valid.jsonl
data/student_guard/classification_test.jsonl

data/student_guard/preference_train.jsonl
data/student_guard/preference_valid.jsonl
data/student_guard/preference_test.jsonl

data/student_guard/eval_examples_train.jsonl
data/student_guard/eval_examples_valid.jsonl
data/student_guard/eval_examples_test.jsonl

data/student_guard/dataset_summary.json
```

### 8.3 这些文件分别干什么

#### A. classification

这是最重要的第一版训练集。

每条样本格式大概是：

```json
{
  "id": "...",
  "prompt": "...",
  "response": "...",
  "label": 1
}
```

其中：

- `label=1` 表示 unsafe
- `label=0` 表示 safe

#### B. preference

这个是后续做 DPO / reward model 的保留集：

```json
{
  "prompt": "...",
  "chosen": "safe refusal",
  "rejected": "unsafe local-model output"
}
```

第一版闭环不强制用它训练，但它已经给你准备好了。

#### C. eval_examples

这是按照你当前 repo 原生 schema 写出来的：

```json
{
  "example_id": "...",
  "task_type": "safety_binary",
  "prompt": "...",
  "response_a": "...",
  "label": 1
}
```

这样你训练完 student guard 以后，可以直接接回 `sentry_jury` 继续评测。

---

## 9. 第四步：训练你的 compact student guard

这一步训练的是 **本地安全分类器**。

### 9.1 第一版最推荐的模型

如果你是新手，我建议优先从下面两种里选一个：

#### 方案 A：最稳最省心

```text
distilroberta-base
```

优点：

- 训练便宜
- 代码最稳
- 很适合先把闭环跑通

#### 方案 B：更贴近你“小模型 student judge”的感觉

```text
Qwen/Qwen2.5-1.5B-Instruct
```

优点：

- 更像 LLM student
- 后面更容易继续做 LoRA / DPO 扩展

但比 DistilRoBERTa 吃资源更多。

---

## 10. 第五步：先跑最稳的分类训练

### 10.1 用 DistilRoBERTa 起步（推荐第一跑）

```bash
python training/train_guard_classifier.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path distilroberta-base \
  --output-dir checkpoints/student_guard_distilroberta
```

### 10.2 如果你想跑更像 LLM 的 student，用 Qwen

```bash
python training/train_guard_classifier.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir checkpoints/student_guard_qwen \
  --trust-remote-code
```

### 10.3 如果你 GPU 不大，想上 LoRA + 4bit

```bash
python training/train_guard_classifier.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir checkpoints/student_guard_qwen_lora \
  --trust-remote-code \
  --use-lora \
  --load-in-4bit
```

### 10.4 训练完成后你会得到什么

```text
checkpoints/student_guard_xxx/
  config.json / adapter_config.json
  tokenizer files
  metrics.json
  label_map.json
```

如果你用了 LoRA，保存出来的大概率是 adapter。
如果没用 LoRA，保存出来就是完整 sequence classification model。

---

## 11. 第六步：单独评估 student guard

先不要急着接回原仓库，先单独测一下 student guard 自己。

### 11.1 直接评估命令

```bash
python scripts/evaluate_student_guard.py \
  --model checkpoints/student_guard_distilroberta \
  --input-jsonl data/student_guard/eval_examples_test.jsonl \
  --output-dir runs/student_guard_direct_eval
```

如果你训练的是 Qwen LoRA 版本：

```bash
python scripts/evaluate_student_guard.py \
  --model checkpoints/student_guard_qwen_lora \
  --input-jsonl data/student_guard/eval_examples_test.jsonl \
  --output-dir runs/student_guard_direct_eval_qwen \
  --trust-remote-code \
  --load-in-4bit
```

### 11.2 你会看到什么

输出目录里会有：

```text
runs/student_guard_direct_eval/predictions.jsonl
runs/student_guard_direct_eval/summary.json
```

重点看：

- `accuracy`
- `precision_unsafe`
- `recall_unsafe`
- `f1_unsafe`
- `unsafe_fnr`

其中 `unsafe_fnr` 越低越好。

---

## 12. 第七步：把 student guard 接回原 repo 继续跑

这一点非常重要，因为这是你项目的亮点：

> 训练出来的 student guard 不是孤立模型，而是重新接回 SENTRY-Jury 框架，成为一个本地 judge backend。

### 12.1 使用新的 YAML 配置

你已经拿到了：

```text
configs/student_guard_eval.yaml
```

你只需要把里面的模型路径改成你自己的 checkpoint：

```yaml
model: checkpoints/student_guard_distilroberta
```

或者：

```yaml
model: checkpoints/student_guard_qwen_lora
```

### 12.2 运行原 repo 风格评测

```bash
python -m sentry_jury.cli --config configs/student_guard_eval.yaml
```

### 12.3 这一步会做什么

它会把 student guard 当成一个正常 judge，继续跑：

- clean evaluation
- attacked evaluation
- probes
- coverage / abstention
- ASR / flip rate

这样你的结果就能和原 repo 风格完全对齐。

---

## 13. 第八步：你后面还能做什么扩展

现在这套补丁的**主线**是：

- teacher mining
- classification student guard
- 接回原框架评测

但我已经给你留了扩展口。

### 13.1 你已经拥有 preference 数据

```text
preference_train.jsonl
preference_valid.jsonl
preference_test.jsonl
```

所以后面你可以继续做：

- reward model
- pairwise scorer
- DPO

### 13.2 我建议的顺序

**第一阶段先别做 DPO。**

先完成这条最稳的线：

```text
teacher -> mined failures -> classification student guard -> repo evaluation
```

等这个跑通以后，再考虑：

```text
preference_train.jsonl -> reward model / ranking model -> 更高级 student
```

---

## 14. 你最容易犯错的地方

### 错误 1：直接在 `main` 上乱改

一定先切到 `boyu`：

```bash
git switch -c boyu origin/boyu
```

### 错误 2：没有先装原 repo 依赖

一定按顺序：

```bash
pip install -r requirements.txt
pip install -e .
pip install -r training/requirements-guard.txt
```

### 错误 3：把 `teacher-config` 配成你还跑不通的 YAML

如果 `configs/api_jailjudge_id.yaml` 依赖外部 API，而你暂时没配好，你就先不要用它做 teacher。
你可以先做一个你自己可跑通的 teacher config，再传给：

```bash
--teacher-config your_config.yaml
```

### 错误 4：一上来就用太大的模型

新手建议先用：

```text
distilroberta-base
```

把全流程跑通以后，再切到：

```text
Qwen/Qwen2.5-1.5B-Instruct + LoRA
```

### 错误 5：只用 mined unsafe，不加 safe

我给你的 `build_guard_datasets.py` 已经自动加入：

- mined unsafe
- template chosen safe
- repo safe

不要自己把 safe 去掉，不然模型非常容易过拒。

---

## 15. 最短可运行版本（你就照这个跑）

如果你现在只想最快跑通，照下面这 6 条命令走。

### Step 1

```bash
git fetch origin
git switch -c boyu origin/boyu
```

### Step 2

把补丁代码复制进 repo。

### Step 3

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
pip install -r training/requirements-guard.txt
```

### Step 4

```bash
python scripts/mine_local_failures.py \
  --teacher-config configs/api_jailjudge_id.yaml \
  --input-jsonl data/final/jailjudge_id_calib.jsonl data/final/jailjudge_id_clean_eval.jsonl \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-jsonl data/student_guard/mined_failures.jsonl \
  --prompt-limit 100 \
  --attack-names identity,roleplay,prefix_injection,translation_bridge,obfuscation \
  --max-new-tokens 256 \
  --do-sample
```

### Step 5

```bash
python scripts/build_guard_datasets.py \
  --mined-jsonl data/student_guard/mined_failures.jsonl \
  --extra-safe-jsonl data/final/jailjudge_id_calib.jsonl data/final/jailjudge_id_clean_eval.jsonl \
  --output-dir data/student_guard
```

### Step 6

```bash
python training/train_guard_classifier.py \
  --train-jsonl data/student_guard/classification_train.jsonl \
  --valid-jsonl data/student_guard/classification_valid.jsonl \
  --model-name-or-path distilroberta-base \
  --output-dir checkpoints/student_guard_distilroberta
```

### Step 7

```bash
python scripts/evaluate_student_guard.py \
  --model checkpoints/student_guard_distilroberta \
  --input-jsonl data/student_guard/eval_examples_test.jsonl \
  --output-dir runs/student_guard_direct_eval
```

### Step 8

把 `configs/student_guard_eval.yaml` 里的模型路径改对，然后：

```bash
python -m sentry_jury.cli --config configs/student_guard_eval.yaml
```

---

## 16. 你最终交作业时可以怎么讲

你可以把项目叙述成：

> 我们把原始的 SENTRY-Jury robust judge scaffold 扩展成一个完整的 training loop。系统先使用 SENTRY teacher 挖掘本地小模型在 jailbreak attack 下的高置信失败样本，再把这些 hard negatives 转换成 compact student guard 的训练集，最后把训练好的 student guard 回接到原始评测框架中进行 clean / attacked robustness evaluation。

这会比“我们只是训练了一个分类器”好很多。

---

## 17. 最后一句建议

**第一版先跑通 classifier 路线。**

不要第一天就上：

- full RLHF
- PPO
- 大模型全参数微调
- 复杂多阶段 DPO

你现在最稳、最适合 `boyu` 分支的闭环就是：

```text
teacher mining -> student guard classifier -> repo evaluation
```

这条线最清楚，也最容易做出可交、可讲、可复现的结果。

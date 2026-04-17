from datasets import load_dataset

gpt_data = load_dataset("ScalerLab/JudgeBench", split="gpt")
claude_data = load_dataset("ScalerLab/JudgeBench", split="claude")
print(gpt_data[0])
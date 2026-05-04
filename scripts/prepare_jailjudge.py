import json
import random
from pathlib import Path

from datasets import load_dataset


def main():
    random.seed(42)
    output_path = Path("data/toxicchat_300.jsonl")

    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")

    unsafe = []
    safe = []

    for item in dataset:
        prompt = item.get("user_input", "").strip()
        response = item.get("model_output", "").strip()
        toxicity = item.get("toxicity", 0)

        if not prompt or not response:
            continue

        if toxicity == 1:
            unsafe.append({"prompt": prompt, "response": response, "label": 1})
        else:
            safe.append({"prompt": prompt, "response": response, "label": -1})

    random.shuffle(unsafe)
    random.shuffle(safe)

    n = min(150, len(unsafe), len(safe))
    selected = unsafe[:n] + safe[:n]
    random.shuffle(selected)

    examples = []
    for i, item in enumerate(selected):
        examples.append({
            "example_id": f"tc{i+1}",
            "task_type": "safety_binary",
            "prompt": item["prompt"],
            "response_a": item["response"],
            "label": item["label"],
        })

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved {len(examples)} examples to {output_path}")
    print(f"unsafe={sum(1 for e in examples if e['label']==1)}, safe={sum(1 for e in examples if e['label']==-1)}")


if __name__ == "__main__":
    main()
from datasets import load_dataset

ds = load_dataset("PKU-Alignment/BeaverTails", split="30k_test")
print(ds[0].keys())
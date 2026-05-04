from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="usail-hkust/JailJudge",
    repo_type="dataset",
)
print(local_dir)
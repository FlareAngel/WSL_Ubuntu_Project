# downloads.py
from huggingface_hub import snapshot_download
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

snapshot_download(
    repo_id="Qwen/Qwen2.5-3B-Instruct",
    local_dir="/mnt/f/Ubuntu/models/Qwen2.5-3B-Instruct",
    local_dir_use_symlinks=False,
    token="hf_你的完整token"   # 替换成你的真实 token
)

# python -m huggingface_hub.commands.huggingface_cli login --token your-token

print("✅ 下载完成")
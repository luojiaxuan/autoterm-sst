"""Fetch the models and evaluation inputs; every repository is public."""
from huggingface_hub import snapshot_download
snapshot_download("gavinlaw/rasst-speech-llm-zh-cap16-denoise-ttag", local_dir="/data/models/zh")
snapshot_download("gavinlaw/rasst-retriever-hn1024", local_dir="/data/models/retriever")
snapshot_download("BAAI/bge-m3", local_dir="/data/models/bge-m3")
snapshot_download("Qwen/Qwen3-4B-Instruct-2507", local_dir="/data/models/router-lm")
snapshot_download("gavinlaw/run-inputs-20261120", repo_type="dataset", local_dir="/data/inputs")
snapshot_download("gavinlaw/rasst-demo-acl6060-zh-segments", repo_type="dataset", local_dir="/data/inputs/acl6060_zh_segments")
snapshot_download("gavinlaw/rasst-main-result-data", repo_type="dataset", local_dir="/data/inputs/main",
                  allow_patterns=["main_result/audio/medicine/*"])
print("downloaded", flush=True)

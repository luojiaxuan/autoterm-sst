"""Fetch the models and evaluation inputs; every repository is public."""
from huggingface_hub import snapshot_download
snapshot_download("gavinlaw/rasst-speech-llm-zh-cap16-denoise-ttag", local_dir="/data/models/zh",
                  revision="5c9558cc1333b847efc8f43c2c505084b8512680")
snapshot_download("gavinlaw/rasst-retriever-hn1024", local_dir="/data/models/retriever")
snapshot_download("BAAI/bge-m3", local_dir="/data/models/bge-m3")
snapshot_download("Qwen/Qwen3-4B-Instruct-2507", local_dir="/data/models/router-lm")
snapshot_download("gavinlaw/run-inputs-20261120", repo_type="dataset", local_dir="/data/inputs",
                  revision="a73684f2d0c546b206b7f5aa3657b63c1417d3b1")
snapshot_download("gavinlaw/rasst-demo-acl6060-zh-segments", repo_type="dataset", local_dir="/data/inputs/acl6060_zh_segments",
                  revision="a10a6a9bab8758bdc50607b50c62f060ed623109")
snapshot_download("gavinlaw/rasst-main-result-data", repo_type="dataset", local_dir="/data/inputs/main",
                  allow_patterns=["main_result/audio/medicine/*"], revision="96b44797d8926d14ef7535e831b861b6af05a28e")
# The RASST retriever loads these by repository id from the Hugging Face cache.
for repo, patterns in (("Atotti/Qwen3-Omni-AudioTransformer", None), ("BAAI/bge-m3", None),
                       ("openai/whisper-large-v3", ["*.json", "*.txt"])):
    print(repo, snapshot_download(repo, allow_patterns=patterns), flush=True)
print("downloaded", flush=True)

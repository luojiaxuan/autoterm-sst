"""Fetch the public models and evaluation inputs. The ESO medicine lectures are prepared separately, as
scripts/radix_stream_eval/download.py describes, and main_radix_arms.sh waits for them."""
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
# The RASST retriever loads these by repository id from the Hugging Face cache.
for repo, patterns in (("Atotti/Qwen3-Omni-AudioTransformer", None), ("BAAI/bge-m3", None),
                       ("openai/whisper-large-v3", ["*.json", "*.txt"])):
    print(repo, snapshot_download(repo, allow_patterns=patterns), flush=True)
print("downloaded", flush=True)

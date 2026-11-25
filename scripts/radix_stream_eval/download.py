"""Fetch the public models and evaluation inputs, and check that the ESO medicine lectures are in place.

The ESO lectures are not redistributed. Download eso_v1.0.tar.gz from the MLLP group
(https://github.com/mllpresearch/ESO-dataset; https://www.mllp.upv.es/eso-dataset/eso_v1.0.tar.gz, 41 GiB,
SHA-256 24529aa4641cdc8b13f235f4fc2a295ab46e159fdaf846e561504ebacd498c6a), convert samples 404, 606, 545006,
596001 and 605000 to 16 kHz mono 16-bit WAV, and save each as
/data/inputs/main/main_result/audio/medicine/sample_<id>_v2/<id>_v2.wav before running the job.
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ESO_SAMPLES = ("404", "606", "545006", "596001", "605000")
MEDICINE = Path("/data/inputs/main/main_result/audio/medicine")

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
missing = [s for s in ESO_SAMPLES if not (MEDICINE / f"sample_{s}_v2" / f"{s}_v2.wav").is_file()]
if missing:
    raise SystemExit(f"ESO lectures {', '.join(missing)} are missing under {MEDICINE}; see this file's docstring")
print("downloaded", flush=True)

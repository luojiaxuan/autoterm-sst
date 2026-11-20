#!/usr/bin/env bash
# Chunk latency of the language-model router, the similarity router and Known-domain-1k, measured one
# configuration at a time on three A6000 GPUs of the Taurus hold job: vllm serve on two GPUs, the retriever
# and the router models on the third. concurrency_full_pipeline.py streams real lectures in real time
# (one 1.92 s chunk every 1.92 s) at 1, 8 and 16 concurrent sessions and records, per chunk, the time from
# sending its audio to receiving its translation. A configuration whose output exists is skipped.
set -euo pipefail
JOB=48285
ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
IO_ROOT=/home/jiaxuanluo/autoterm_taurus_20261024_io
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
OUT="${IO_ROOT}/runs/latency_20261120"
mkdir -p "${OUT}"
cd "${ROOT}/code"
git rev-parse HEAD > "${OUT}/code_revision.txt"

# The capacity server is built from an explicit config, so its vllm serve URL is the default port 8200.
measure() {  # measure <mode> <preset>
  local mode=$1 preset=$2 port=8040 vport=8200
  [[ -s "${OUT}/${mode}.json" ]] && return 0
  tmux new-session -d -s lat-server \
    "srun --jobid=${JOB} --overlap --gres=gpu:3 --ntasks=1 --cpus-per-task=12 --mem=80G env CONTEXT_DEVICE=cuda:0 VLLM_UTIL=0.85 ${ROOT}/code/scripts/taurus_evidence_server_vllm_serve.sh ${port} ${vport} ${mode} > ${OUT}/server_${mode}.log 2>&1"
  for _ in $(seq 1 400); do
    curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 && break
    tmux has-session -t =lat-server
    sleep 3
  done
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" > "${OUT}/health_${mode}.json"
  "${PYTHON_BIN}" - "${OUT}/health_${mode}.json" "${mode}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["rag"]["status"] == "ready", d["rag"]
assert d["backend"].startswith("vllm_serve"), d["backend"]
if sys.argv[2] != "fixed":
    assert d["context_similarity"]["status"] == "ready", d["context_similarity"]
PY
  local granted
  granted=$(head -1 "${OUT}/server_${mode}.log" | sed 's/.*CUDA_VISIBLE_DEVICES=//')
  srun --jobid=${JOB} --overlap --gres=none --ntasks=1 --cpus-per-task=4 --mem=16G "${PYTHON_BIN}" -u \
    eval/streaming_sst/concurrency_full_pipeline.py --base-url "http://127.0.0.1:${port}" --preset "${preset}" \
    --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/acl6060/*.wav' \
    --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine/sample_*_v2/*.wav' \
    --levels 1,8,16 --seconds-per-session 180 --gpus "${granted}" \
    --out-json "${OUT}/${mode}.json" > "${OUT}/client_${mode}.log" 2>&1
  tmux kill-session -t =lat-server
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${granted}" | awk '{s+=$1} END {print s}')
    [[ "${used}" -lt 100 ]] && break
    sleep 5
  done
}

measure lm auto_working
measure routed auto_working
measure fixed nlp_core_1k
touch "${OUT}/done"

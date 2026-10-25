#!/usr/bin/env bash
# note (luojiaxuan): After the vllm serve G1 finishes, rerun it with the BGE-M3 router model on GPU and 12 CPUs.
set -euo pipefail

JOB=48285
ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
IO_ROOT=/home/jiaxuanluo/autoterm_taurus_20261024_io
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python

while tmux has-session -t sst-chain 2>/dev/null; do sleep 20; done
GPUS=$(head -1 "${IO_ROOT}/logs/server_8034.log" | sed 's/.*CUDA_VISIBLE_DEVICES=//')
tmux kill-session -t sst-c2
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPUS}" | awk '{s+=$1} END {print s}')
  [[ "${used}" -lt 100 ]] && break
  sleep 2
done

tmux new-session -d -s sst-c3 \
  "srun --jobid=${JOB} --overlap --gres=gpu:2 --ntasks=1 --cpus-per-task=12 --mem=60G env CONTEXT_DEVICE=cuda:0 ${ROOT}/code/scripts/taurus_evidence_server_vllm_serve.sh 8035 8200 > ${IO_ROOT}/logs/server_8035.log 2>&1"
for _ in $(seq 1 400); do
  curl -fsS --max-time 3 http://127.0.0.1:8035/health >/dev/null 2>&1 && break
  tmux has-session -t sst-c3 2>/dev/null
  sleep 3
done
curl -fsS --max-time 3 http://127.0.0.1:8035/health | "${PYTHON_BIN}" -c "
import json, sys
d = json.load(sys.stdin)
assert d['rag']['status'] == 'ready', d['rag']
assert d['context_similarity']['status'] == 'ready', d['context_similarity']
assert d['backend'].startswith('vllm_serve'), d['backend']
print('8035 ready', d['backend'])
"
GPUS2=$(head -1 "${IO_ROOT}/logs/server_8035.log" | sed 's/.*CUDA_VISIBLE_DEVICES=//')
cd "${ROOT}/code"
srun --jobid=${JOB} --overlap --gres=none --ntasks=1 --cpus-per-task=4 --mem=16G "${PYTHON_BIN}" -u eval/streaming_sst/concurrency_full_pipeline.py \
  --base-url http://127.0.0.1:8035 \
  --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/acl6060/*.wav' \
  --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine/sample_*_v2/*.wav' \
  --levels 1,8,16,32 --seconds-per-session 180 --gpus "${GPUS2}" \
  --out-json "${IO_ROOT}/runs/g1_vllm_serve_gpu_context.json" > "${IO_ROOT}/logs/g1_vllm_serve_gpu_context.log" 2>&1
echo "g1c done"

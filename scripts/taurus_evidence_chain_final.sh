#!/usr/bin/env bash
# note (luojiaxuan): After the three baseline runs finish: score them, free server B, and run the
# note (luojiaxuan): dedicated-retrieval G1 on three GPUs of allocation 48285 that are physically idle.
set -euo pipefail

JOB=48285
ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
IO_ROOT=/home/jiaxuanluo/autoterm_taurus_20261024_io
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
CLIENT="srun --jobid=${JOB} --overlap --gres=none --ntasks=1"

for run in merged10k_taurus merged2k_taurus known_taurus; do
  until [[ -s "${IO_ROOT}/runs/${run}.json" ]]; do sleep 30; done
done
while tmux has-session -t =sst-eb 2>/dev/null; do sleep 10; done
for run in merged10k_taurus merged2k_taurus known_taurus; do
  ${CLIENT} --cpus-per-task=2 --mem=8G "${ROOT}/code/scripts/taurus_evidence_score.sh" "${run}" 2>&1 | tail -2
done
echo "scoring done"

GPUS_B=$(head -1 "${IO_ROOT}/logs/server_8032.log" | sed 's/.*CUDA_VISIBLE_DEVICES=//')
tmux kill-session -t =sst-b
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPUS_B}" | awk '{s+=$1} END {print s}')
  [[ "${used}" -lt 100 ]] && break
  sleep 2
done

# note (luojiaxuan): Slurm cannot see processes started outside it, so pick GPUs by actual memory use.
FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0,1,2,3,4,5 | awk -F', ' '$2 < 100 {print $1}' | head -3 | paste -sd, -)
[[ $(echo "${FREE}" | tr ',' '\n' | grep -c .) -eq 3 ]]
echo "dedicated-retrieval G1 on GPUs ${FREE}"

tmux new-session -d -s sst-c5 \
  "srun --jobid=${JOB} --overlap --gres=none --ntasks=1 --cpus-per-task=12 --mem=60G env CUDA_VISIBLE_DEVICES=${FREE} CONTEXT_DEVICE=cuda:0 VLLM_UTIL=0.85 ${ROOT}/code/scripts/taurus_evidence_server_vllm_serve.sh 8037 8200 > ${IO_ROOT}/logs/server_8037.log 2>&1"
for _ in $(seq 1 400); do
  curl -fsS --max-time 3 http://127.0.0.1:8037/health >/dev/null 2>&1 && break
  tmux has-session -t =sst-c5 2>/dev/null
  sleep 3
done
curl -fsS --max-time 3 http://127.0.0.1:8037/health | "${PYTHON_BIN}" -c "
import json, sys
d = json.load(sys.stdin)
assert d['rag']['status'] == 'ready', d['rag']
assert d['context_similarity']['status'] == 'ready', d['context_similarity']
assert d['backend'].startswith('vllm_serve'), d['backend']
print('8037 ready', d['backend'])
"
cd "${ROOT}/code"
${CLIENT} --cpus-per-task=4 --mem=16G "${PYTHON_BIN}" -u eval/streaming_sst/concurrency_full_pipeline.py \
  --base-url http://127.0.0.1:8037 \
  --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/acl6060/*.wav' \
  --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine/sample_*_v2/*.wav' \
  --levels 1,8,16,24,32 --seconds-per-session 180 --gpus "${FREE}" \
  --out-json "${IO_ROOT}/runs/g1_vllm_serve_dedicated_retrieval.json" > "${IO_ROOT}/logs/g1_vllm_serve_dedicated_retrieval.log" 2>&1
tmux kill-session -t =sst-c5
echo "final chain done"

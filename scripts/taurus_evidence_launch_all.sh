#!/usr/bin/env bash
# note (luojiaxuan): Start the three evidence servers and all clients as Slurm steps of allocation 48285.
# note (luojiaxuan): Every step is held by its own tmux session; logs and outputs go to the home disk.
set -euo pipefail

JOB=48285
ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
IO_ROOT=/home/jiaxuanluo/autoterm_taurus_20261024_io
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
mkdir -p "${IO_ROOT}/logs" "${IO_ROOT}/runs"

for session in sst-g1 sst-ea sst-eb1 sst-eb23 sst-a sst-b sst-c; do
  tmux kill-session -t "${session}" 2>/dev/null || true
done
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2,3,4,5 | awk '{s+=$1} END {print s}')
  [[ "${used}" -lt 300 ]] && break
  sleep 2
done

server() {
  local session="$1" port="$2" mode="$3" seqs="$4" util="$5" cpus="$6"
  tmux new-session -d -s "${session}" \
    "srun --jobid=${JOB} --overlap --gres=gpu:2 --ntasks=1 --cpus-per-task=${cpus} --mem=60G ${ROOT}/code/scripts/taurus_evidence_server.sh ${port} ${mode} ${seqs} ${util} > ${IO_ROOT}/logs/server_${port}.log 2>&1"
}
server sst-c 8033 routed 32 0.78 8
server sst-a 8031 routed 8 0.72 12
server sst-b 8032 fixed 8 0.72 12

for port in 8031 8032 8033; do
  for _ in $(seq 1 300); do
    if curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      break
    fi
    sleep 3
  done
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/health" | "${PYTHON_BIN}" -c "
import json, sys
d = json.load(sys.stdin)
assert d['rag']['status'] == 'ready', d['rag']
if ${port} != 8032:
    assert d['context_similarity']['status'] == 'ready', d['context_similarity']
print(${port}, 'ready')
"
done

CLIENT="srun --jobid=${JOB} --overlap --gres=none --ntasks=1"
tmux new-session -d -s sst-ea \
  "${CLIENT} --cpus-per-task=2 --mem=12G ${ROOT}/code/scripts/taurus_evidence_eval.sh 8031 auto_working autoterm_taurus > ${IO_ROOT}/logs/eval_autoterm_taurus.log 2>&1"
tmux new-session -d -s sst-eb \
  "${CLIENT} --cpus-per-task=2 --mem=24G bash -c '${ROOT}/code/scripts/taurus_evidence_eval.sh 8032 merged_10x1k merged10k_taurus > ${IO_ROOT}/logs/eval_merged10k_taurus.log 2>&1 & ${ROOT}/code/scripts/taurus_evidence_eval.sh 8032 merged_nlp_med_2k merged2k_taurus > ${IO_ROOT}/logs/eval_merged2k_taurus.log 2>&1 & ${ROOT}/code/scripts/taurus_evidence_eval.sh 8032 nlp_core_1k known_taurus oracle > ${IO_ROOT}/logs/eval_known_taurus.log 2>&1 & wait'"
tmux new-session -d -s sst-g1 \
  "cd ${ROOT}/code && ${CLIENT} --cpus-per-task=4 --mem=16G ${PYTHON_BIN} -u eval/streaming_sst/concurrency_full_pipeline.py --base-url http://127.0.0.1:8033 --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/acl6060/*.wav' --audio-glob '/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine/sample_*_v2/*.wav' --levels 1,8,16,32 --seconds-per-session 180 --gpus \$(tail -n +1 ${IO_ROOT}/logs/server_8033.log | head -1 | sed 's/.*CUDA_VISIBLE_DEVICES=//') --out-json ${IO_ROOT}/runs/g1_full_pipeline.json > ${IO_ROOT}/logs/g1_full_pipeline.log 2>&1"
sleep 5
tmux ls
squeue -s -j "${JOB}"
date -u +%FT%TZ

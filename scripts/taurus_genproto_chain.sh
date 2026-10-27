#!/usr/bin/env bash
# note (luojiaxuan): Closed-loop AutoTerm rerun with router texts generated from each slice glossary
# note (luojiaxuan): (configs/router_prototypes_generated_s0.json), otherwise identical to the sst-a AutoTerm run.
# note (luojiaxuan): Waits for two physically idle GPUs of allocation 48285, serves, streams the ten-talk playlist,
# note (luojiaxuan): stops the server and scores the run. Writes runs/<name>.done on success.
set -euo pipefail

JOB=48285
NAME=autoterm_genproto_s0_taurus
PORT=8041
ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
IO_ROOT=/home/jiaxuanluo/autoterm_taurus_20261024_io
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
PROTOTYPES="${ROOT}/code/configs/router_prototypes_generated_s0.json"
CLIENT="srun --jobid=${JOB} --overlap --gres=none --ntasks=1"
mkdir -p "${IO_ROOT}/logs" "${IO_ROOT}/runs"

idle_gpus() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0,1,2,3,4,5 \
    | awk -F', ' '$2 < 100 {print $1}' | head -2 | paste -sd, -
}

# note (luojiaxuan): Slurm cannot see processes started outside it, so take two cards only when they are idle
# note (luojiaxuan): on two readings a minute apart.
while true; do
  first=$(idle_gpus)
  sleep 60
  second=$(idle_gpus)
  if [[ "${first}" == "${second}" && $(echo "${second}" | tr ',' '\n' | grep -c .) -eq 2 ]]; then
    break
  fi
  echo "$(date -u +%FT%TZ) waiting, idle=${second:-none}"
  sleep 540
done
GPUS="${second}"
echo "$(date -u +%FT%TZ) launching on GPUs ${GPUS}"

tmux new-session -d -s sst-gp \
  "${CLIENT} --cpus-per-task=12 --mem=60G env CUDA_VISIBLE_DEVICES=${GPUS} ${ROOT}/code/scripts/taurus_evidence_server.sh ${PORT} routed 8 0.72 ${PROTOTYPES} > ${IO_ROOT}/logs/server_${PORT}.log 2>&1"
for _ in $(seq 1 400); do
  curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  tmux has-session -t =sst-gp 2>/dev/null
  sleep 3
done
curl -fsS --max-time 3 "http://127.0.0.1:${PORT}/health" | "${PYTHON_BIN}" -c "
import json, sys
d = json.load(sys.stdin)
assert d['rag']['status'] == 'ready', d['rag']
cs = d['context_similarity']
assert cs['status'] == 'ready', cs
expected = json.load(open('${PROTOTYPES}', encoding='utf-8'))
assert cs['prototypes'] == expected, cs['prototypes']
print('${PORT} ready with generated prototypes for', sorted(expected))
"

${CLIENT} --cpus-per-task=2 --mem=12G "${ROOT}/code/scripts/taurus_evidence_eval.sh" "${PORT}" auto_working "${NAME}" \
  > "${IO_ROOT}/logs/eval_${NAME}.log" 2>&1
tmux kill-session -t =sst-gp
${CLIENT} --cpus-per-task=2 --mem=8G "${ROOT}/code/scripts/taurus_evidence_score.sh" "${NAME}" 2>&1 | tail -2
date -u +%FT%TZ > "${IO_ROOT}/runs/${NAME}.done"
echo "genproto chain done"

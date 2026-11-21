#!/usr/bin/env bash
# Noise floor: each merged checkpoint served with eager kernels, scored against its CUDA-graph targets.
set -euo pipefail
cd "$(dirname "$0")"
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
COMMON=(--host 127.0.0.1 --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization "${UTIL:-0.46}"
  --trust-remote-code --enforce-eager)
wait_ready() {
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
run() {  # run <lang> <gpu> <port>
  CUDA_VISIBLE_DEVICES="$2" vllm serve "/data/models/$1" --served-model-name m --port "$3" "${COMMON[@]}" > "serve_eager_$1.log" 2>&1 &
  local pid=$!
  wait_ready "$3" "$pid"
  python3 check.py compare --url "http://127.0.0.1:$3" --model m --lang "$1" --sentences sents.txt --targets "targets_$1.json" --out "result_eager_$1.json"
  kill "$pid"; wait "$pid" || true
}
run zh "${GPUS[0]}" 18700 & A=$!
run de "${GPUS[1]}" 18701 & B=$!
wait "$A" "$B"
run ja "${GPUS[0]}" 18700

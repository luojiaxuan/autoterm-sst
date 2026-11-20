#!/usr/bin/env bash
# Compare base + recovered LoRA against each merged checkpoint under vLLM.
set -euo pipefail
cd "$(dirname "$0")"
M=/data/models
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct", local_dir="/data/models/base")
snapshot_download("gavinlaw/rasst-speech-llm-cap16-denoise-ttag-lora", local_dir="/data/models/lora")
for l in ("zh", "de", "ja"):
    snapshot_download(f"gavinlaw/rasst-speech-llm-{l}-cap16-denoise-ttag", local_dir=f"/data/models/{l}")
PY
PORT=18700
COMMON=(--port "$PORT" --host 127.0.0.1 --tensor-parallel-size 2 --max-model-len 4096 --gpu-memory-utilization "${UTIL:-0.40}" --trust-remote-code)
wait_ready() {
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && return 0
    kill -0 "$1" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
for l in zh de ja; do
  vllm serve "$M/$l" --served-model-name m "${COMMON[@]}" > "serve_merged_$l.log" 2>&1 &
  PID=$!
  wait_ready "$PID"
  python3 check.py generate --url "http://127.0.0.1:$PORT" --model m --lang "$l" --sentences sents.txt --targets "targets_$l.json"
  kill "$PID"; wait "$PID" || true
done
vllm serve "$M/base" --served-model-name base "${COMMON[@]}" \
  --enable-lora --enable-mixed-moe-lora-format --max-lora-rank 32 --max-loras 3 \
  --lora-modules \
    '{"name": "zh", "path": "/data/models/lora/zh", "is_3d_lora_weight": false}' \
    '{"name": "de", "path": "/data/models/lora/de", "is_3d_lora_weight": false}' \
    '{"name": "ja", "path": "/data/models/lora/ja", "is_3d_lora_weight": false}' \
  > serve_lora.log 2>&1 &
PID=$!
wait_ready "$PID"
for l in zh de ja; do
  python3 check.py compare --url "http://127.0.0.1:$PORT" --model "$l" --lang "$l" --sentences sents.txt --targets "targets_$l.json" --out "result_$l.json"
done
kill "$PID"; wait "$PID" || true

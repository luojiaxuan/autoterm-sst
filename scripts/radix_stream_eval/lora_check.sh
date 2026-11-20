#!/usr/bin/env bash
# Compare base + recovered LoRA against each merged checkpoint under vLLM, one GPU per server.
# The first GPU serves each merged checkpoint in turn and records greedy targets; the second serves the
# base model with the three adapters and scores those targets.
set -euo pipefail
cd "$(dirname "$0")"
M=/data/models
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-Omni-30B-A3B-Instruct", local_dir="/data/models/base")
snapshot_download("gavinlaw/rasst-speech-llm-cap16-denoise-ttag-lora", local_dir="/data/models/lora",
                  revision="a82e03699c697b52a69e5702094fb4a7a81aeb52")
for l in ("zh", "de", "ja"):
    snapshot_download(f"gavinlaw/rasst-speech-llm-{l}-cap16-denoise-ttag", local_dir=f"/data/models/{l}")
PY
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
COMMON=(--host 127.0.0.1 --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization "${UTIL:-0.46}"
  --trust-remote-code)
wait_ready() {  # wait_ready <port> <pid>
  for _ in $(seq 1 360); do
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null && return 0
    kill -0 "$2" 2>/dev/null || return 1
    sleep 5
  done
  return 1
}
CUDA_VISIBLE_DEVICES="${GPUS[1]}" vllm serve "$M/base" --served-model-name base --port 18701 "${COMMON[@]}" \
  --enable-lora --enable-mixed-moe-lora-format --max-lora-rank 32 --max-loras 3 \
  --lora-modules \
    '{"name": "zh", "path": "/data/models/lora/zh", "is_3d_lora_weight": false}' \
    '{"name": "de", "path": "/data/models/lora/de", "is_3d_lora_weight": false}' \
    '{"name": "ja", "path": "/data/models/lora/ja", "is_3d_lora_weight": false}' \
  > serve_lora.log 2>&1 &
LORA=$!
for l in zh de ja; do
  if [[ ! -s "targets_$l.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" vllm serve "$M/$l" --served-model-name m --port 18700 "${COMMON[@]}" \
      > "serve_merged_$l.log" 2>&1 &
    PID=$!
    wait_ready 18700 "$PID"
    python3 check.py generate --url http://127.0.0.1:18700 --model m --lang "$l" --sentences sents.txt --targets "targets_$l.json"
    kill "$PID"; wait "$PID" || true
  fi
done
wait_ready 18701 "$LORA"
for l in zh de ja; do
  python3 check.py compare --url http://127.0.0.1:18701 --model "$l" --lang "$l" --sentences sents.txt --targets "targets_$l.json" --out "result_$l.json"
done
kill "$LORA"; wait "$LORA" || true

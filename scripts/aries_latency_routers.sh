#!/usr/bin/env bash
# Chunk latency of the language-model router, the similarity router and Known-domain-1k on three idle A6000
# GPUs of aries, borrowed outside Slurm. Code, models and slices come from the Taurus NFS mounts through the
# directory $BASE, whose entries link to them under neutral names. Each configuration is served alone with
# vllm serve on the first two GPUs and the retriever and router models on the third, and
# concurrency_full_pipeline.py streams real lectures in real time ($LEVELS sessions, $SESSION_SEC seconds each,
# 1, 8 and 16 sessions of 180 s by default, as in taurus_latency_routers.sh). A configuration whose output
# exists is skipped.
set -euo pipefail
BASE=${BASE:-/mnt/data6/jiaxuanluo/sglang-omni-lat}
GPUS=${GPUS:-0,1,2}
OUT=${OUT:?set OUT to the output directory}
LEVELS=${LEVELS:-1,8,16}
SESSION_SEC=${SESSION_SEC:-180}
WARMUP_SEC=${WARMUP_SEC:-$SESSION_SEC}
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
export ROOT="$BASE" IO_ROOT="$BASE/io" MODEL="$BASE/model" RETRIEVER="$BASE/retriever.pt"
export MANIFEST="$BASE/manifest.json" RETRIEVER_CODE="$BASE/retriever_code"
mkdir -p "$OUT"
cd "$BASE/code"
git rev-parse HEAD > "$OUT/code_revision.txt"
# Read the model once over NFS so that every server start loads it from the page cache.
cat "$MODEL"/*.safetensors > /dev/null

measure() {  # measure <mode> <preset>
  local mode=$1 preset=$2 port=8040 pid used
  [[ -s "$OUT/$mode.json" ]] && return 0
  CUDA_VISIBLE_DEVICES="$GPUS" CONTEXT_DEVICE=cuda:0 VLLM_UTIL=0.85 \
    setsid "$BASE/code/scripts/taurus_evidence_server_vllm_serve.sh" "$port" 8200 "$mode" > "$OUT/server_$mode.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 600); do
    curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
    kill -0 "$pid"
    sleep 3
  done
  curl -fsS --max-time 3 "http://127.0.0.1:$port/health" > "$OUT/health_$mode.json"
  "$PYTHON_BIN" - "$OUT/health_$mode.json" "$mode" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["rag"]["status"] == "ready", d["rag"]
assert d["backend"].startswith("vllm_serve"), d["backend"]
if sys.argv[2] != "fixed":
    assert d["context_similarity"]["status"] == "ready", d["context_similarity"]
PY
  CUDA_VISIBLE_DEVICES= "$PYTHON_BIN" -u eval/streaming_sst/concurrency_full_pipeline.py \
    --base-url "http://127.0.0.1:$port" --preset "$preset" \
    --audio-glob "$BASE/audio_acl/*.wav" --audio-glob "$BASE/audio_med/sample_*_v2/*.wav" \
    --levels "$LEVELS" --seconds-per-session "$SESSION_SEC" --warmup-seconds "$WARMUP_SEC" --gpus "$GPUS" \
    --out-json "$OUT/$mode.json" > "$OUT/client_$mode.log" 2>&1
  kill -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | awk '{s+=$1} END {print s}')
    [[ "$used" -lt 100 ]] && break
    sleep 5
  done
}

measure lm auto_working
measure routed auto_working
measure fixed nlp_core_1k
touch "$OUT/done"

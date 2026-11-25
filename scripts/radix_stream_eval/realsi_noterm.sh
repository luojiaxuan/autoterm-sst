#!/usr/bin/env bash
# No-glossary reference on the RealSI streams: the fixed-glossary server of realsi.sh with a retrieval
# threshold above any MaxSim score, so no term pair enters the prompt and every chunk gets the empty term
# block. The three orders run as concurrent sessions. A stream whose output exists is skipped.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p code out logs tmp
tar xzf code.tar.gz -C code
tar xzf rasst.tar.gz
export RASST_ACTIVE_CODE_ROOT="$HERE/rasst"
TF=$(python3 -c "import transformers; print(transformers.__version__)")
pip install -q peft soundfile librosa audioread av qwen-omni-utils faiss-cpu "transformers==$TF" > logs/pip.log 2>&1
pip freeze > logs/pip_freeze.txt
SLICES_OUT=/data/inputs/realsi_slices
if [[ ! -s "$SLICES_OUT/manifest.json" ]]; then
  python3 code/scripts/build_realsi_slices.py --slices /data/inputs/slices --terms /data/inputs/realsi/realsi_terms.json \
    --out "$SLICES_OUT" --node-root "$SLICES_OUT" --retriever-dir "$HERE/rasst/retriever" \
    --model-path /data/models/retriever/hn1024.pt > logs/build_slices.log 2>&1
fi
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
UTIL=${UTIL:-0.31}
BASE=(--host 127.0.0.1 --model-path /data/models/zh --manifest "$SLICES_OUT/manifest.json"
  --rag-model-path /data/models/retriever/hn1024.pt --rag-device cuda:0
  --max-num-seqs 32 --max-model-len 16384
  --vllm-limit-audio 16 --vllm-enforce-eager 1 --enable-prefix-caching 1 --disable-custom-all-reduce 1
  --vllm-use-v1 1 --vllm-enable-v1-multiprocessing 1 --vllm-worker-multiproc-method spawn
  --vllm-moe-use-deep-gemm 0 --vllm-use-fused-moe-grouped-topk 0 --nccl-p2p-disable 1 --nccl-ib-disable 1
  --torch-nccl-enable-monitoring 0 --extra-python-path "$HERE/rasst/retriever"
  --scheduler-batch-size 32 --max-inflight-batches 2 --max-new-tokens 40 --rag-top-k 10
  --rag-score-threshold 0.78 --retrieval-candidate-budget 100 --term-map-format tagged
  --empty-term-map-policy none_block --log-level info --vllm-tp-size "${#GPUS[@]}" --gpu-memory-utilization "$UTIL")
ORDERS=(1 2 3)
cd code
export HF_HUB_OFFLINE=1
client() {  # client <port> <preset> <name> <order> [extra args]
  local port=$1 preset=$2 name=$3 order=$4; shift 4
  python3 -u eval/streaming_sst/eval_mixed_audio_switch.py --base-url "http://127.0.0.1:$port" \
    --playlist "/data/inputs/realsi/playlist_s$order.json" --preset "$preset" --language-pair 'English -> Chinese' \
    --latency-multiplier 2 --base-segment-sec 0.96 --feed-sleep 0 --max-unacked-chunks 1 \
    --backpressure-stall-timeout-sec 60 --idle-timeout-sec 5 --idle-timeouts-after-eof 1 --max-switch-events 20 \
    --no-assert "$@" --out-json "../out/${name}_s$order.json" --out-md "../out/${name}_s$order.md" \
    > "../logs/client_${name}_s$order.log" 2>&1
}
server() {  # server <port> <name> <routed|fixed> [args]: start, wait for /health, check retrieval and routing
  local port=$1 name=$2 kind=$3; shift 3
  python3 -u eval/streaming_sst/serve_glossary_capacity_sweep.py --port "$port" "${BASE[@]}" \
    --tmp-dir "$HERE/tmp/$name" "$@" > "../logs/server_$name.log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 720); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$port/health" > "../logs/health_$name.json"
  python3 - "../logs/health_$name.json" "$kind" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["rag"]["status"] == "ready", d["rag"]
if sys.argv[2] == "routed":
    assert d["context_similarity"]["status"] == "ready", d["context_similarity"]
PY
}
stop() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
echo "gpus=${CUDA_VISIBLE_DEVICES} util=$UTIL" > ../logs/plan_noterm.txt
server 18903 noterm fixed --required-presets nlp_core_1k --autoterm-policy fixed --rag-score-threshold 1.01
pids=()
for o in "${ORDERS[@]}"; do
  if [[ ! -s "../out/noterm_s$o.json" ]]; then
    client 18903 nlp_core_1k noterm "$o" --allow-missing-router-meta &
    pids+=($!)
  fi
done
[[ ${#pids[@]} -eq 0 ]] || wait "${pids[@]}"
stop
touch ../out/done_noterm

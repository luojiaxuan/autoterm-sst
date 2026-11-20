#!/usr/bin/env bash
# Ten-domain RealSI streams: the ten RealSI en2zh talks played back to back in three orders. The slices
# first gain each talk's annotated terms (build_realsi_slices.py); both routers then read the generated
# prototypes. Servers run one at a time over all granted GPUs: the language-model router, the similarity
# router, then the fixed glossaries serving Known-domain-1k and Merged-10k as two concurrent sessions.
# Each server takes the three orders in turn. A stream whose output exists is skipped, so the script resumes.
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
SLICES=nlp_core_1k,medicine_core_1k,education_core_1k,finance_core_1k,legal_core_1k,environment_core_1k,entertainment_core_1k,science_core_1k,sports_core_1k,art_core_1k
PROTOTYPES="$HERE/code/configs/router_prototypes_generated_s0.json"
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
ROUTED=(--required-presets "$SLICES" --autoterm-policy budgeted_top_slices --autoterm-default-preset nlp_core_1k
  --autoterm-term-budget 4000 --autoterm-max-active-slices 4 --autoterm-context-window-chunks 4
  --autoterm-context-model /data/models/bge-m3 --autoterm-context-device cuda:0 --autoterm-context-prototypes "$PROTOTYPES"
  --autoterm-context-weight 1.0 --autoterm-text-weight 0.0 --autoterm-domain-probe-weight 0.0
  --autoterm-speech-weight 0.0 --autoterm-metadata-weight 0.0 --autoterm-ema-alpha 0.80
  --autoterm-update-sec 5.0 --autoterm-warmup-sec 6.0 --autoterm-cooldown-sec 30.0)
LM=(--autoterm-context-scorer language_model --autoterm-context-language-model /data/models/router-lm
  --autoterm-switch-rate 1e-3 --no-autoterm-pin-active-slice)
FIXED=(--required-presets "$SLICES,merged_10x1k" --autoterm-policy fixed)
ORACLE=nlp=nlp_core_1k,medicine=medicine_core_1k,education=education_core_1k,finance=finance_core_1k,legal=legal_core_1k,environment=environment_core_1k,entertainment=entertainment_core_1k,science=science_core_1k,sports=sports_core_1k,art=art_core_1k
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
stop() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; sleep 20; }
todo() { for o in "${ORDERS[@]}"; do for n in "$@"; do [[ -s "../out/${n}_s$o.json" ]] || return 0; done; done; return 1; }
echo "gpus=${CUDA_VISIBLE_DEVICES} util=$UTIL" > ../logs/plan.txt
if todo lm_hmm; then
  server 18900 lm_hmm routed "${ROUTED[@]}" "${LM[@]}"
  for o in "${ORDERS[@]}"; do [[ -s "../out/lm_hmm_s$o.json" ]] || client 18900 auto_working lm_hmm "$o"; done
  stop
fi
if todo cosine; then
  server 18901 cosine routed "${ROUTED[@]}"
  for o in "${ORDERS[@]}"; do [[ -s "../out/cosine_s$o.json" ]] || client 18901 auto_working cosine "$o"; done
  stop
fi
if todo known merged10k; then
  server 18902 fixed fixed "${FIXED[@]}"
  for o in "${ORDERS[@]}"; do
    [[ -s "../out/known_s$o.json" ]] || client 18902 nlp_core_1k known "$o" --allow-missing-router-meta \
      --oracle-preset-map "$ORACLE" &
    [[ -s "../out/merged10k_s$o.json" ]] || client 18902 merged_10x1k merged10k "$o" --allow-missing-router-meta &
    wait
  done
  stop
fi
touch ../out/done

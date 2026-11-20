#!/usr/bin/env bash
# Ten-talk stream, one server at a time over all granted GPUs with tensor parallelism. Phase lm_hmm runs the
# language-model router with the HMM filter, phase cosine the similarity router, and phase fixed serves the
# fixed glossaries to Known-domain-1k, Merged-2k and Merged-10k as three concurrent sessions. A phase whose
# outputs exist is skipped, so the script resumes after an interruption.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p code out logs tmp
tar xzf code.tar.gz -C code
tar xzf retriever.tar.gz -C code
cp /data/inputs/slices/manifest.json code/manifest.json
# The image's vLLM needs its own Transformers v5, so it is held fixed while the rest is installed.
TF=$(python3 -c "import transformers; print(transformers.__version__)")
pip install -q peft soundfile librosa audioread av qwen-omni-utils faiss-cpu "transformers==$TF" > logs/pip.log 2>&1
pip freeze > logs/pip_freeze.txt
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
UTIL=${UTIL:-0.33}
SLICES=nlp_core_1k,medicine_core_1k,education_core_1k,finance_core_1k,legal_core_1k,environment_core_1k,entertainment_core_1k,science_core_1k,sports_core_1k,art_core_1k
BASE=(--host 127.0.0.1 --model-path /data/models/zh --manifest "$HERE/code/manifest.json"
  --rag-model-path /data/models/retriever/hn1024.pt --rag-device cuda:0
  --max-num-seqs 32 --max-model-len 16384
  --vllm-limit-audio 16 --vllm-enforce-eager 1 --enable-prefix-caching 1 --disable-custom-all-reduce 1
  --vllm-use-v1 1 --vllm-enable-v1-multiprocessing 1 --vllm-worker-multiproc-method spawn
  --vllm-moe-use-deep-gemm 0 --vllm-use-fused-moe-grouped-topk 0 --nccl-p2p-disable 1 --nccl-ib-disable 1
  --torch-nccl-enable-monitoring 0 --extra-python-path "$HERE/code/retriever"
  --scheduler-batch-size 32 --max-inflight-batches 2 --max-new-tokens 40 --rag-top-k 10
  --rag-score-threshold 0.78 --retrieval-candidate-budget 100 --term-map-format tagged
  --empty-term-map-policy none_block --log-level info --vllm-tp-size "${#GPUS[@]}" --gpu-memory-utilization "$UTIL")
ROUTED=(--required-presets "$SLICES" --autoterm-policy budgeted_top_slices --autoterm-default-preset nlp_core_1k
  --autoterm-term-budget 4000 --autoterm-max-active-slices 4 --autoterm-context-window-chunks 4
  --autoterm-context-model /data/models/bge-m3 --autoterm-context-device cuda:0
  --autoterm-context-weight 1.0 --autoterm-text-weight 0.0 --autoterm-domain-probe-weight 0.0
  --autoterm-speech-weight 0.0 --autoterm-metadata-weight 0.0 --autoterm-ema-alpha 0.80
  --autoterm-update-sec 5.0 --autoterm-warmup-sec 6.0 --autoterm-cooldown-sec 30.0)
LM=(--autoterm-context-scorer language_model --autoterm-context-language-model /data/models/router-lm
  --autoterm-switch-rate 1e-3 --no-autoterm-pin-active-slice)
FIXED=(--required-presets merged_10x1k,merged_nlp_med_2k,nlp_core_1k,medicine_core_1k --autoterm-policy fixed)
cd code
export HF_HUB_OFFLINE=1
serve() {  # serve <port> <name> [policy args]
  local port=$1 name=$2; shift 2
  python3 -u eval/streaming_sst/serve_glossary_capacity_sweep.py --port "$port" "${BASE[@]}" \
    --tmp-dir "$HERE/tmp/$name" "$@" > "../logs/server_$name.log" 2>&1 &
  echo $!
}
ready() {
  for _ in $(seq 1 720); do curl -sf "http://127.0.0.1:$1/health" >/dev/null && return 0; sleep 5; done
  return 1
}
client() {  # client <port> <preset> <name> [extra args]
  local port=$1 preset=$2 name=$3; shift 3
  python3 -u eval/streaming_sst/eval_mixed_audio_switch.py --base-url "http://127.0.0.1:$port" \
    --acl-root /data/inputs/acl6060_zh_segments --medicine-audio-dir /data/inputs/main/main_result/audio/medicine \
    --acl-items 5 --medicine-items 5 --medicine-ids 404,606,545006,596001,605000 --schedule alternating \
    --seed 20260707 --preset "$preset" --language-pair 'English -> Chinese' --latency-multiplier 2 \
    --base-segment-sec 0.96 --feed-sleep 0 --max-unacked-chunks 1 --backpressure-stall-timeout-sec 60 \
    --idle-timeout-sec 5 --idle-timeouts-after-eof 1 --max-switch-events 20 --no-assert "$@" \
    --out-json "../out/$name.json" --out-md "../out/$name.md" > "../logs/client_$name.log" 2>&1
}
echo "gpus=${CUDA_VISIBLE_DEVICES} util=$UTIL" > ../logs/plan.txt
phase() {  # phase <port> <name> <server args...> -- <client spec "preset name [args]">...
  local port=$1 name=$2; shift 2
  local args=() specs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do args+=("$1"); shift; done
  shift
  specs=("$@")
  local todo=0
  for spec in "${specs[@]}"; do read -r -a f <<< "$spec"; [[ -s "../out/${f[1]}.json" ]] || todo=1; done
  [[ $todo -eq 1 ]] || return 0
  local pid pids=()
  pid=$(serve "$port" "$name" "${args[@]}")
  ready "$port"
  for spec in "${specs[@]}"; do read -r -a f <<< "$spec"; client "$port" "${f[@]}" & pids+=($!); done
  wait "${pids[@]}" || true
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 20
}
phase 18800 lm_hmm "${ROUTED[@]}" "${LM[@]}" -- "auto_working autoterm_lm_hmm"
phase 18801 cosine "${ROUTED[@]}" -- "auto_working autoterm_cosine"
phase 18802 fixed "${FIXED[@]}" -- \
  "nlp_core_1k known --allow-missing-router-meta --oracle-preset-map nlp=nlp_core_1k,medicine=medicine_core_1k" \
  "merged_nlp_med_2k merged2k --allow-missing-router-meta" \
  "merged_10x1k merged10k --allow-missing-router-meta"
touch ../out/done

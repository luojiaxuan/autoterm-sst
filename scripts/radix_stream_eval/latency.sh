#!/usr/bin/env bash
# Latency and processing speed on exclusive GPUs, one single-GPU server per configuration: the
# language-model router, the similarity router, and Known-domain-1k. Each server first serves one
# real-time session over the first 180 s of an oncology lecture (chunk latency, time to first text),
# then the full ten-talk stream (processing time over audio duration). The three servers run side by
# side on their own GPUs. A step whose output exists is skipped, so the script resumes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p code out logs tmp
tar xzf code.tar.gz -C code
tar xzf rasst.tar.gz
export RASST_ACTIVE_CODE_ROOT="$HERE/rasst"
cp /data/inputs/slices/manifest.json code/manifest.json
sed -i -E 's#[^" ]*/acl6060_zh_segments/#/data/inputs/acl6060_zh_segments/#g' \
  /data/inputs/acl6060_zh_segments/segments.meta.jsonl /data/inputs/acl6060_zh_segments/audio.yaml
TF=$(python3 -c "import transformers; print(transformers.__version__)")
pip install -q peft soundfile librosa audioread av qwen-omni-utils faiss-cpu "transformers==$TF" > logs/pip.log 2>&1
pip freeze > logs/pip_freeze.txt
nvidia-smi > logs/nvidia_smi_start.txt
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES}"
UTIL=${UTIL:-0.80}
SLICES=nlp_core_1k,medicine_core_1k,education_core_1k,finance_core_1k,legal_core_1k,environment_core_1k,entertainment_core_1k,science_core_1k,sports_core_1k,art_core_1k
BASE=(--host 127.0.0.1 --model-path /data/models/zh --manifest "$HERE/code/manifest.json"
  --rag-model-path /data/models/retriever/hn1024.pt --rag-device cuda:0
  --max-num-seqs 32 --max-model-len 16384
  --vllm-limit-audio 16 --vllm-enforce-eager 1 --enable-prefix-caching 1 --disable-custom-all-reduce 1
  --vllm-use-v1 1 --vllm-enable-v1-multiprocessing 1 --vllm-worker-multiproc-method spawn
  --vllm-moe-use-deep-gemm 0 --vllm-use-fused-moe-grouped-topk 0 --nccl-p2p-disable 1 --nccl-ib-disable 1
  --torch-nccl-enable-monitoring 0 --extra-python-path "$HERE/rasst/retriever"
  --scheduler-batch-size 32 --max-inflight-batches 2 --max-new-tokens 40 --rag-top-k 10
  --rag-score-threshold 0.78 --retrieval-candidate-budget 100 --term-map-format tagged
  --empty-term-map-policy none_block --log-level info --vllm-tp-size 1 --gpu-memory-utilization "$UTIL")
ROUTED=(--required-presets "$SLICES" --autoterm-policy budgeted_top_slices --autoterm-default-preset nlp_core_1k
  --autoterm-term-budget 4000 --autoterm-max-active-slices 4 --autoterm-context-window-chunks 4
  --autoterm-context-model /data/models/bge-m3 --autoterm-context-device cuda:0
  --autoterm-context-weight 1.0 --autoterm-text-weight 0.0 --autoterm-domain-probe-weight 0.0
  --autoterm-speech-weight 0.0 --autoterm-metadata-weight 0.0 --autoterm-ema-alpha 0.80
  --autoterm-update-sec 5.0 --autoterm-warmup-sec 6.0 --autoterm-cooldown-sec 30.0)
LM=(--autoterm-context-scorer language_model --autoterm-context-language-model /data/models/router-lm
  --autoterm-switch-rate 1e-3 --no-autoterm-pin-active-slice)
FIXED=(--required-presets nlp_core_1k,medicine_core_1k --autoterm-policy fixed)
cd code
export HF_HUB_OFFLINE=1

measure() {  # measure <gpu index> <port> <name> <preset> <routed|fixed> <stream client args> -- <server args...>
  local gpu=$1 port=$2 name=$3 preset=$4 kind=$5 extra=$6; shift 7
  if [[ -s "../out/stream_$name.json" ]]; then return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u eval/streaming_sst/serve_glossary_capacity_sweep.py --port "$port" \
    "${BASE[@]}" --tmp-dir "$HERE/tmp/$name" "$@" > "../logs/server_$name.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 720); do curl -sf "http://127.0.0.1:$port/health" >/dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$port/health" > "../logs/health_$name.json"
  python3 - "../logs/health_$name.json" "$kind" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["rag"]["status"] == "ready", d["rag"]
if sys.argv[2] == "routed":
    assert d["context_similarity"]["status"] == "ready", d["context_similarity"]
PY
  if [[ ! -s "../out/session_$name.json" ]]; then
    python3 -u eval/streaming_sst/concurrency_full_pipeline.py --base-url "http://127.0.0.1:$port" \
      --preset "$preset" --audio-glob '/data/inputs/main/main_result/audio/medicine/sample_404_v2/*.wav' \
      --levels 1 --seconds-per-session 180 --gpus "$gpu" \
      --out-json "../out/session_$name.json" > "../logs/session_$name.log" 2>&1
  fi
  # shellcheck disable=SC2086
  python3 -u eval/streaming_sst/eval_mixed_audio_switch.py --base-url "http://127.0.0.1:$port" \
    --acl-root /data/inputs/acl6060_zh_segments --medicine-audio-dir /data/inputs/main/main_result/audio/medicine \
    --acl-items 5 --medicine-items 5 --medicine-ids 404,606,545006,596001,605000 --schedule alternating \
    --seed 20260707 --preset "$preset" --language-pair 'English -> Chinese' --latency-multiplier 2 \
    --base-segment-sec 0.96 --feed-sleep 0 --max-unacked-chunks 1 --backpressure-stall-timeout-sec 60 \
    --idle-timeout-sec 5 --idle-timeouts-after-eof 1 --max-switch-events 20 --no-assert $extra \
    --out-json "../out/stream_$name.json" --out-md "../out/stream_$name.md" > "../logs/stream_$name.log" 2>&1
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

echo "gpus=${CUDA_VISIBLE_DEVICES} util=$UTIL" > ../logs/plan.txt
measure "${GPUS[0]}" 18810 lm_hmm auto_working routed "" -- "${ROUTED[@]}" "${LM[@]}" &
P0=$!
measure "${GPUS[1]}" 18811 cosine auto_working routed "" -- "${ROUTED[@]}" &
P1=$!
measure "${GPUS[2]}" 18812 known nlp_core_1k fixed \
  "--allow-missing-router-meta --oracle-preset-map nlp=nlp_core_1k,medicine=medicine_core_1k" -- "${FIXED[@]}" &
P2=$!
wait $P0 $P1 $P2
nvidia-smi > ../logs/nvidia_smi_end.txt
touch ../out/done

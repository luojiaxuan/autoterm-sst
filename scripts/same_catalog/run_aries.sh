#!/usr/bin/env bash
# The ten-talk stream over one catalog scale on three A6000 GPUs of a host without containers. Arms, flags
# and completeness checks follow run.sh: the routed arm on its own server, then the merged and known-domain
# arms, one after another, on a fixed-policy server; only a complete arm reaches out/ and is skipped on
# resume. Layout as in the Taurus evidence runs: in-process vLLM with tensor parallelism on the first two
# visible GPUs, and the retriever, BGE-M3 and the router model on the third.
#   run_aries.sh <100k|1m> <scale dir>
# Environment: CODE RASST PYHOOK PY MODEL RETRIEVER BGE ROUTER_LM BUNDLE REFERENCE ACL MEDICINE TMP_ROOT GPUS,
# VECTOR_SOURCES (space-separated GPU-built indexes that hold every catalog term's vector),
# CUDA_VISIBLE_DEVICES (three GPUs), optional GPU_UTIL (0.85) and PORT (18820; the fixed server takes PORT+1).
set -euo pipefail
SCALE=${1:?scale}
HERE=${2:?scale dir}
: "${CODE:?}" "${RASST:?}" "${PYHOOK:?}" "${PY:?}" "${MODEL:?}" "${RETRIEVER:?}" "${BGE:?}" "${ROUTER_LM:?}"
: "${BUNDLE:?}" "${REFERENCE:?}" "${ACL:?}" "${MEDICINE:?}" "${TMP_ROOT:?}" "${GPUS:?}" "${VECTOR_SOURCES:?}"
read -r -a SOURCES <<< "$VECTOR_SOURCES"
GPU_UTIL=${GPU_UTIL:-0.85}
mkdir -p "$HERE/out" "$HERE/logs" "$HERE/tmp"
cd "$HERE"
export RASST_ACTIVE_CODE_ROOT="$RASST"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PYHOOK:$CODE"
export UPSTREAM_COMPAT_HOOK="$CODE/serve/vllm_compat/sitecustomize.py"
export TMPDIR="$TMP_ROOT/$SCALE"
mkdir -p "$TMPDIR"

(while sleep 60; do printf '%s %s\n' "$(date -u +%FT%TZ)" "$(cat logs/stage 2>/dev/null)" > heartbeat; done) &
BEAT=$!
SERVER=
stop() {
  [[ -n "$SERVER" ]] || return 0
  kill -TERM -- "-$SERVER" 2>/dev/null || true
  for _ in $(seq 1 30); do kill -0 -- "-$SERVER" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SERVER" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=
  sleep 20
}
cleanup() {
  kill "$BEAT" 2>/dev/null || true
  stop
}
trap cleanup EXIT
trap 'exit 143' TERM INT

CAT="$HERE/catalog"
echo build > logs/stage
[[ -s "$CAT/report.json" ]] || "$PY" -u "$CODE/scripts/same_catalog/build.py" --scale "$SCALE" --out "$CAT" \
  --bundle "$BUNDLE" --reference "$REFERENCE" --retriever-code "$RASST/retriever" --retriever "$RETRIEVER" \
  --context-model "$BGE" --text-model "$BGE" --vector-sources "${SOURCES[@]}" --layout balanced --device cuda:0 \
  > logs/build.log 2>&1
BUDGET=$("$PY" -S -c "import json; print(json.load(open('$CAT/report.json'))['budget'])")
SLICES=nlp_core_10k,medicine_core_10k,education_core_10k,finance_core_10k,legal_core_10k,environment_core_10k,entertainment_core_10k,science_core_10k,sports_core_10k,art_core_10k
BASE=(--host 127.0.0.1 --model-path "$MODEL" --manifest "$CAT/manifest.json"
  --rag-model-path "$RETRIEVER" --rag-device cuda:2
  --max-num-seqs 32 --max-model-len 16384
  --vllm-limit-audio 16 --vllm-enforce-eager 1 --enable-prefix-caching 1 --disable-custom-all-reduce 1
  --vllm-use-v1 1 --vllm-enable-v1-multiprocessing 1 --vllm-worker-multiproc-method spawn
  --vllm-moe-use-deep-gemm 0 --vllm-use-fused-moe-grouped-topk 0 --nccl-p2p-disable 1 --nccl-ib-disable 1
  --torch-nccl-enable-monitoring 0 --extra-python-path "$RASST/retriever" --vllm-compat-dir "$PYHOOK"
  --scheduler-batch-size 32 --max-inflight-batches 2 --max-new-tokens 40 --rag-top-k 10
  --rag-score-threshold 0.78 --retrieval-candidate-budget 100 --reference-ranking-policy dense_pair
  --term-map-format tagged
  --empty-term-map-policy none_block --log-level info --vllm-tp-size 2 --gpu-memory-utilization "$GPU_UTIL")
ROUTED=(--required-presets "$SLICES" --autoterm-policy budgeted_top_slices --autoterm-default-preset nlp_core_10k
  --autoterm-term-budget "$BUDGET" --autoterm-max-active-slices 4 --autoterm-context-window-chunks 4
  --autoterm-context-model "$BGE" --autoterm-context-device cuda:2
  --autoterm-context-weight 1.0 --autoterm-text-weight 0.0 --autoterm-domain-probe-weight 0.0
  --autoterm-speech-weight 0.0 --autoterm-metadata-weight 0.0 --autoterm-ema-alpha 0.80
  --autoterm-update-sec 5.0 --autoterm-warmup-sec 6.0 --autoterm-cooldown-sec 30.0
  --autoterm-context-scorer language_model --autoterm-context-language-model "$ROUTER_LM"
  --autoterm-switch-rate 1e-3 --no-autoterm-pin-active-slice)
FIXED=(--required-presets merged_all,nlp_core_10k,medicine_core_10k --autoterm-policy fixed)

"$PY" -S - "$SCALE" "$GPU_UTIL" "$BUDGET" "$CODE/REVISION" "$GPUS" > manifest.json <<'PY'
import json, os, subprocess, sys, time
print(json.dumps({
    "started": time.strftime("%FT%TZ", time.gmtime()), "scale": sys.argv[1], "vllm_utilization": sys.argv[2],
    "term_budget": sys.argv[3], "host": os.uname().nodename, "revision": open(sys.argv[4]).read().strip(),
    "layout": "vLLM tensor parallel 2 on the first two GPUs; retriever, BGE-M3 and router model on the third",
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader",
                                    "--id=" + sys.argv[5]], text=True).strip().splitlines(),
    "heartbeat": os.path.abspath("heartbeat"), "stall_threshold_sec": 1200,
    "restart": "rerun this script; finished arms are skipped", "stop": "touch stop in the run root",
}, indent=1))
PY

cd "$CODE"
ready() {  # ready <port> <routed|fixed>
  for _ in $(seq 1 360); do
    kill -0 "$SERVER" || return 1
    curl --noproxy '*' -sf "http://127.0.0.1:$1/health" > /dev/null && break
    sleep 5
  done
  curl --noproxy '*' -sf "http://127.0.0.1:$1/health" > "$HERE/logs/health_$2.json"
  "$PY" -S - "$HERE/logs/health_$2.json" "$2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["rag"]["status"] == "ready", d["rag"]
if sys.argv[2] == "routed":
    assert d["context_similarity"]["status"] == "ready", d["context_similarity"]
    assert len(d["context_similarity"]["domains"]) == 10, d["context_similarity"]
PY
}
client() {  # client <port> <preset> <name> [extra args]
  local port=$1 preset=$2 name=$3
  shift 3
  "$PY" -u eval/streaming_sst/eval_mixed_audio_switch.py --base-url "http://127.0.0.1:$port" \
    --acl-root "$ACL" --medicine-audio-dir "$MEDICINE" \
    --acl-items 5 --medicine-items 5 --medicine-ids 404,606,545006,596001,605000 --schedule alternating \
    --seed 20260707 --preset "$preset" --language-pair 'English -> Chinese' --latency-multiplier 2 \
    --base-segment-sec 0.96 --feed-sleep 0 --max-unacked-chunks 1 --backpressure-stall-timeout-sec 60 \
    --idle-timeout-sec 5 --idle-timeouts-after-eof 1 --max-switch-events 20 --no-assert "$@" \
    --out-json "$HERE/tmp/$name.json" --out-md "$HERE/out/$name.md" > "$HERE/logs/client_$name.log" 2>&1
  "$PY" -S - "$HERE/tmp/$name.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["pacing"]
assert len(d["records"]) >= 8776, len(d["records"])
assert p["sent_samples"] == p["acknowledged_cursor_samples"] == 269569844, p
assert p["timeout_release_count"] == 0, p
PY
  mv "$HERE/tmp/$name.json" "$HERE/out/$name.json"
}
serve() {  # serve <port> <routed|fixed> [policy args]
  local port=$1 name=$2
  shift 2
  echo "$name" > "$HERE/logs/stage"
  setsid "$PY" -u eval/streaming_sst/serve_glossary_capacity_sweep.py --port "$port" "${BASE[@]}" \
    --tmp-dir "$TMPDIR/$name" "$@" > "$HERE/logs/server_$name.log" 2>&1 &
  SERVER=$!
  ready "$port" "$name"
}

PORT=${PORT:-18820}
if [[ ! -s "$HERE/out/routed.json" ]]; then
  serve "$PORT" routed "${ROUTED[@]}"
  client "$PORT" auto_working routed
  stop
fi
PORT=$((PORT + 1))
if [[ ! -s "$HERE/out/merged.json" || ! -s "$HERE/out/known.json" ]]; then
  serve "$PORT" fixed "${FIXED[@]}"
  if [[ ! -s "$HERE/out/merged.json" ]]; then
    echo merged > "$HERE/logs/stage"
    client "$PORT" merged_all merged --allow-missing-router-meta
  fi
  if [[ ! -s "$HERE/out/known.json" ]]; then
    echo known > "$HERE/logs/stage"
    client "$PORT" nlp_core_10k known --allow-missing-router-meta \
      --oracle-preset-map nlp=nlp_core_10k,medicine=medicine_core_10k
  fi
  stop
fi
echo done > "$HERE/logs/stage"
touch "$HERE/out/done"

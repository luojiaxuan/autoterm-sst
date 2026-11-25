#!/usr/bin/env bash
# Ten-talk stream over one catalog scale on one GPU: the routed arm on its own server, then the merged and
# known-domain arms, one after another, on a fixed-policy server. Only a complete arm reaches out/ and
# is skipped on resume.
#   bash run.sh <100k|1m>
set -euo pipefail
SCALE=${1:?scale}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p code out logs tmp
tar xzf code.tar.gz -C code
tar xzf rasst.tar.gz
export RASST_ACTIVE_CODE_ROOT="$HERE/rasst"
export HF_HUB_OFFLINE=1
sed -i -E 's#[^" ]*/acl6060_zh_segments/#/data/inputs/acl6060_zh_segments/#g' \
  /data/inputs/acl6060_zh_segments/segments.meta.jsonl /data/inputs/acl6060_zh_segments/audio.yaml
TF=$(python3 -c "import transformers; print(transformers.__version__)")
pip install -q setproctitle peft soundfile librosa audioread av qwen-omni-utils faiss-cpu "transformers==$TF" > logs/pip.log 2>&1
pip freeze > logs/pip_freeze.txt
export PYTHONPATH="$HERE/pyhook:$HERE/code${PYTHONPATH:+:$PYTHONPATH}"

(while sleep 60; do printf '%s %s\n' "$(date -u +%FT%TZ)" "$(cat logs/stage 2>/dev/null)" > heartbeat; done) &
BEAT=$!
SERVER=
cleanup() {
  kill "$BEAT" 2>/dev/null || true
  [[ -n "$SERVER" ]] && kill "$SERVER" 2>/dev/null || true
}
trap cleanup EXIT

CAT="$HERE/catalog"
echo build > logs/stage
[[ -s "$CAT/report.json" ]] || python3 -u code/scripts/same_catalog/build.py --scale "$SCALE" --out "$CAT" \
  --bundle /data/inputs/bundle --reference /data/inputs/slices --retriever-code "$HERE/rasst/retriever" \
  --retriever /data/models/retriever/hn1024.pt --context-model /data/models/bge-m3 --layout balanced > logs/build.log 2>&1
BUDGET=$(python3 -c "import json; print(json.load(open('$CAT/report.json'))['budget'])")
# The share of the card that leaves vLLM VLLM_GB.
UTIL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id="$CUDA_VISIBLE_DEVICES" |
  awk -v gb="${VLLM_GB:-82}" '{printf "%.3f", gb * 1024 / $1}')
SLICES=nlp_core_10k,medicine_core_10k,education_core_10k,finance_core_10k,legal_core_10k,environment_core_10k,entertainment_core_10k,science_core_10k,sports_core_10k,art_core_10k
BASE=(--host 127.0.0.1 --model-path /data/models/zh --manifest "$CAT/manifest.json"
  --rag-model-path /data/models/retriever/hn1024.pt --rag-device cuda:0
  --max-num-seqs 32 --max-model-len 16384
  --vllm-limit-audio 16 --vllm-enforce-eager 1 --enable-prefix-caching 1 --disable-custom-all-reduce 1
  --vllm-use-v1 1 --vllm-enable-v1-multiprocessing 1 --vllm-worker-multiproc-method spawn
  --vllm-moe-use-deep-gemm 0 --vllm-use-fused-moe-grouped-topk 0 --nccl-p2p-disable 1 --nccl-ib-disable 1
  --torch-nccl-enable-monitoring 0 --extra-python-path "$HERE/rasst/retriever"
  --scheduler-batch-size 32 --max-inflight-batches 2 --max-new-tokens 40 --rag-top-k 10
  --rag-score-threshold 0.78 --retrieval-candidate-budget 100 --reference-ranking-policy dense_pair
  --term-map-format tagged
  --empty-term-map-policy none_block --log-level info --vllm-tp-size 1 --gpu-memory-utilization "$UTIL")
ROUTED=(--required-presets "$SLICES" --autoterm-policy budgeted_top_slices --autoterm-default-preset nlp_core_10k
  --autoterm-term-budget "$BUDGET" --autoterm-max-active-slices 4 --autoterm-context-window-chunks 4
  --autoterm-context-model /data/models/bge-m3 --autoterm-context-device cuda:0
  --autoterm-context-weight 1.0 --autoterm-text-weight 0.0 --autoterm-domain-probe-weight 0.0
  --autoterm-speech-weight 0.0 --autoterm-metadata-weight 0.0 --autoterm-ema-alpha 0.80
  --autoterm-update-sec 5.0 --autoterm-warmup-sec 6.0 --autoterm-cooldown-sec 30.0
  --autoterm-context-scorer language_model --autoterm-context-language-model /data/models/router-lm
  --autoterm-switch-rate 1e-3 --no-autoterm-pin-active-slice)
FIXED=(--required-presets merged_all,nlp_core_10k,medicine_core_10k --autoterm-policy fixed)

python3 - "$SCALE" "$UTIL" "$BUDGET" > manifest.json <<'PY'
import json, os, subprocess, sys, time
print(json.dumps({
    "started": time.strftime("%FT%TZ", time.gmtime()), "scale": sys.argv[1], "vllm_utilization": sys.argv[2],
    "term_budget": sys.argv[3], "host": os.uname().nodename, "revision": open("code/REVISION").read().strip(),
    "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader",
                                    "--id=" + os.environ["CUDA_VISIBLE_DEVICES"]], text=True).strip(),
    "heartbeat": os.path.abspath("heartbeat"), "stall_threshold_sec": 1200,
    "restart": "rerun run.sh; finished arms are skipped", "stop": "teardown the container",
}, indent=1))
PY

cd code
ready() {  # ready <port> <routed|fixed>
  for _ in $(seq 1 360); do
    kill -0 "$SERVER" || return 1
    curl -sf "http://127.0.0.1:$1/health" > /dev/null && break
    sleep 5
  done
  curl -sf "http://127.0.0.1:$1/health" > "../logs/health_$2.json"
  python3 - "../logs/health_$2.json" "$2" <<'PY'
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
  python3 -u eval/streaming_sst/eval_mixed_audio_switch.py --base-url "http://127.0.0.1:$port" \
    --acl-root /data/inputs/acl6060_zh_segments --medicine-audio-dir /data/inputs/main/main_result/audio/medicine \
    --acl-items 5 --medicine-items 5 --medicine-ids 404,606,545006,596001,605000 --schedule alternating \
    --seed 20260707 --preset "$preset" --language-pair 'English -> Chinese' --latency-multiplier 2 \
    --base-segment-sec 0.96 --feed-sleep 0 --max-unacked-chunks 1 --backpressure-stall-timeout-sec 60 \
    --idle-timeout-sec 5 --idle-timeouts-after-eof 1 --max-switch-events 20 --no-assert "$@" \
    --out-json "../tmp/$name.json" --out-md "../out/$name.md" > "../logs/client_$name.log" 2>&1
  python3 - "../tmp/$name.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["pacing"]
assert len(d["records"]) >= 8776, len(d["records"])
assert p["sent_samples"] == p["acknowledged_cursor_samples"] == 269569844, p
assert p["timeout_release_count"] == 0, p
PY
  mv "../tmp/$name.json" "../out/$name.json"
}
serve() {  # serve <port> <routed|fixed> [policy args]
  local port=$1 name=$2
  shift 2
  echo "$name" > ../logs/stage
  python3 -u eval/streaming_sst/serve_glossary_capacity_sweep.py --port "$port" "${BASE[@]}" \
    --tmp-dir "$HERE/tmp/$name" "$@" > "../logs/server_$name.log" 2>&1 &
  SERVER=$!
  ready "$port" "$name"
}
stop() {
  kill "$SERVER"
  wait "$SERVER" || true
  SERVER=
  sleep 20
}

PORT=${PORT:-18820}
if [[ ! -s ../out/routed.json ]]; then
  serve "$PORT" routed "${ROUTED[@]}"
  client "$PORT" auto_working routed
  stop
fi
PORT=$((PORT + 1))
if [[ ! -s ../out/merged.json || ! -s ../out/known.json ]]; then
  serve "$PORT" fixed "${FIXED[@]}"
  if [[ ! -s ../out/merged.json ]]; then
    echo merged > ../logs/stage
    client "$PORT" merged_all merged --allow-missing-router-meta
  fi
  if [[ ! -s ../out/known.json ]]; then
    echo known > ../logs/stage
    client "$PORT" nlp_core_10k known --allow-missing-router-meta \
      --oracle-preset-map nlp=nlp_core_10k,medicine=medicine_core_10k
  fi
  stop
fi
echo done > ../logs/stage
touch ../out/done

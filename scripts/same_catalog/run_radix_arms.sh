#!/usr/bin/env bash
# The ten-talk stream over one prebuilt catalog: the routed arm on its own server, and the merged and known-domain
# arms as two concurrent sessions of a fixed-policy server. Each server takes two GPUs, as in the H100 concurrency
# runs: the speech LLM alone on the first (tensor parallel 1), the retriever, router model and BGE-M3 on the second.
# With two GPUs the two servers run one after the other, with four at once. Flags and completeness checks follow
# run.sh; only a complete arm reaches out/ and is skipped on resume.
#   bash run_radix_arms.sh <catalog dir>
# Environment: CUDA_VISIBLE_DEVICES (two or four GPUs), optional VLLM_UTIL (0.95) and PORT (18830).
set -euo pipefail
CAT=${1:?catalog dir}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p code out logs tmp
tar xzf code.tar.gz -C code
tar xzf rasst.tar.gz
export RASST_ACTIVE_CODE_ROOT="$HERE/rasst"
export HF_HUB_OFFLINE=1
sed -i -E 's#[^" ]*/acl6060_zh_segments/#/data/inputs/acl6060_zh_segments/#g' \
  /data/inputs/acl6060_zh_segments/segments.meta.jsonl /data/inputs/acl6060_zh_segments/audio.yaml
# The image's vLLM needs its own Transformers v5, so it is held fixed while the rest is installed.
TF=$(python3 -c "import transformers; print(transformers.__version__)")
pip install -q setproctitle peft soundfile librosa audioread av qwen-omni-utils faiss-cpu "transformers==$TF" > logs/pip.log 2>&1
pip freeze > logs/pip_freeze.txt
export PYTHONPATH="$HERE/pyhook:$HERE/code${PYTHONPATH:+:$PYTHONPATH}"

# The catalog was built on another host; its manifest names that host's root.
python3 - "$CAT" <<'PY'
import json, sys
from pathlib import Path
cat = Path(sys.argv[1])
path = cat / "manifest.json"
manifest = json.loads(path.read_text(encoding="utf-8"))
old = manifest["root"]
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1).replace(old, str(cat)), encoding="utf-8")
PY

(while sleep 60; do printf '%s %s\n' "$(date -u +%FT%TZ)" "$(cat logs/stage 2>/dev/null)" > heartbeat; done) &
BEAT=$!
PIDS=()
cleanup() {
  kill "$BEAT" 2>/dev/null || true
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#GPUS[@]} -eq 2 || ${#GPUS[@]} -eq 4 ]] || { echo "two or four GPUs are needed, got $CUDA_VISIBLE_DEVICES"; exit 2; }
BUDGET=$(python3 -c "import json; print(json.load(open('$CAT/report.json'))['budget'])")
DOMAINS=$(python3 -c "import json; print(len(json.load(open('$CAT/report.json'))['slice_pairs']))")
SLICES=$(python3 -c "import json; print(','.join(d + '_core_10k' for d in json.load(open('$CAT/report.json'))['slice_pairs']))")
UTIL=${VLLM_UTIL:-0.95}
BASE=(--host 127.0.0.1 --model-path /data/models/zh --manifest "$CAT/manifest.json"
  --rag-model-path /data/models/retriever/hn1024.pt --rag-device cuda:1
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
  --autoterm-context-model /data/models/bge-m3 --autoterm-context-device cuda:1
  --autoterm-context-weight 1.0 --autoterm-text-weight 0.0 --autoterm-domain-probe-weight 0.0
  --autoterm-speech-weight 0.0 --autoterm-metadata-weight 0.0 --autoterm-ema-alpha 0.80
  --autoterm-update-sec 5.0 --autoterm-warmup-sec 6.0 --autoterm-cooldown-sec 30.0
  --autoterm-context-scorer language_model --autoterm-context-language-model /data/models/router-lm
  --autoterm-context-prototypes "$CAT/prototypes.json"
  --autoterm-switch-rate 1e-3 --no-autoterm-pin-active-slice)
FIXED=(--required-presets merged_all,nlp_core_10k,medicine_core_10k --autoterm-policy fixed)

python3 - "$UTIL" "$BUDGET" "$CAT" > manifest.json <<'PY'
import json, os, subprocess, sys, time
print(json.dumps({
    "started": time.strftime("%FT%TZ", time.gmtime()), "vllm_utilization": sys.argv[1], "term_budget": sys.argv[2],
    "catalog": sys.argv[3], "host": os.uname().nodename, "revision": open("code/REVISION").read().strip(),
    "rasst_patches": open("rasst/PATCHES").read().split(),
    "gpus": subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,uuid", "--format=csv,noheader",
                                     "--id=" + os.environ["CUDA_VISIBLE_DEVICES"]], text=True).strip().splitlines(),
    "layout": "routed server, then a fixed-policy server with the merged and known-domain sessions at once; the "
              "speech LLM alone on a server's first GPU, retriever, router model and BGE-M3 on its second",
    "heartbeat": os.path.abspath("heartbeat"), "stall_threshold_sec": 1200,
    "restart": "rerun main.sh run_radix_arms.sh; finished arms are skipped", "stop": "teardown the container",
}, indent=1))
PY

cd code
client() {  # client <port> <preset> <name> [extra args]
  local port=$1 preset=$2 name=$3
  shift 3
  python3 -u eval/streaming_sst/eval_mixed_audio_switch.py --base-url "http://127.0.0.1:$port" \
    --acl-root /data/inputs/acl6060_zh_segments --medicine-audio-dir /data/inputs/main/main_result/audio/medicine \
    --acl-items 5 --medicine-items 5 --medicine-ids 404,606,545006,596001,605000 --schedule alternating \
    --seed 20260707 --preset "$preset" --language-pair 'English -> Chinese' --latency-multiplier 2 \
    --base-segment-sec 0.96 --feed-sleep 0 --max-unacked-chunks 1 --backpressure-stall-timeout-sec 60 \
    --idle-timeout-sec 5 --idle-timeouts-after-eof 1 --max-switch-events 20 --no-assert "$@" \
    --out-json "../tmp/$name.json" --out-md "../out/$name.md" > "../logs/client_$name.log" 2>&1 || return 1
  # Explicit returns: group may run inside an || list, where set -e does not apply.
  python3 - "../tmp/$name.json" <<'PY2' || return 1
import json, sys
d = json.load(open(sys.argv[1]))
p = d["pacing"]
assert len(d["records"]) >= 8776, len(d["records"])
assert p["sent_samples"] == p["acknowledged_cursor_samples"] == 269569844, p
assert p["timeout_release_count"] == 0, p
PY2
  mv "../tmp/$name.json" "../out/$name.json"
}
group() {  # group <gpu,gpu> <port> <server name> <routed|fixed> -- "<preset> <arm name> [client args]"...
  local gpu=$1 port=$2 name=$3 kind=$4
  shift 5
  local specs=("$@") todo=() spec f
  for spec in "${specs[@]}"; do read -r -a f <<< "$spec"; [[ -s "../out/${f[1]}.json" ]] || todo+=("$spec"); done
  [[ ${#todo[@]} -gt 0 ]] || return 0
  local policy
  if [[ $kind == routed ]]; then policy=("${ROUTED[@]}"); else policy=("${FIXED[@]}"); fi
  CUDA_VISIBLE_DEVICES=$gpu python3 -u eval/streaming_sst/serve_glossary_capacity_sweep.py --port "$port" "${BASE[@]}" \
    --tmp-dir "$HERE/tmp/$name" "${policy[@]}" > "../logs/server_$name.log" 2>&1 &
  local server=$!
  for _ in $(seq 1 720); do
    kill -0 "$server" || { echo "server $name exited"; return 1; }
    curl -sf "http://127.0.0.1:$port/health" > /dev/null && break
    sleep 5
  done
  curl -sf "http://127.0.0.1:$port/health" > "../logs/health_$name.json" || { kill "$server"; return 1; }
  python3 - "../logs/health_$name.json" "$kind" "$DOMAINS" <<'PY2' || { kill "$server"; return 1; }
import json, sys
d = json.load(open(sys.argv[1]))
assert d["rag"]["status"] == "ready", d["rag"]
if sys.argv[2] == "routed":
    assert d["context_similarity"]["status"] == "ready", d["context_similarity"]
    assert len(d["context_similarity"]["domains"]) == int(sys.argv[3]), d["context_similarity"]
PY2
  local pids=() ok=0
  for spec in "${todo[@]}"; do read -r -a f <<< "$spec"; client "$port" "${f[@]}" & pids+=($!); done
  for pid in "${pids[@]}"; do wait "$pid" || ok=1; done
  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
  # The next server on these GPUs starts once the engine process has released its memory.
  for _ in $(seq 1 60); do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="$gpu" | awk '$1 > 2048 {busy=1} END {exit !busy}' || break
    sleep 5
  done
  return $ok
}

PORT=${PORT:-18830}
MERGED="merged_all merged --allow-missing-router-meta"
KNOWN="nlp_core_10k known --allow-missing-router-meta --oracle-preset-map nlp=nlp_core_10k,medicine=medicine_core_10k"
echo arms > ../logs/stage
status=0
if [[ ${#GPUS[@]} -eq 4 ]]; then
  group "${GPUS[0]},${GPUS[1]}" "$PORT" routed routed -- "auto_working routed" & PIDS+=($!)
  group "${GPUS[2]},${GPUS[3]}" $((PORT + 1)) fixed fixed -- "$MERGED" "$KNOWN" & PIDS+=($!)
  for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
  PIDS=()
else
  group "${GPUS[0]},${GPUS[1]}" "$PORT" routed routed -- "auto_working routed" || status=1
  group "${GPUS[0]},${GPUS[1]}" $((PORT + 1)) fixed fixed -- "$MERGED" "$KNOWN" || status=1
fi
[[ $status -eq 0 ]] || { echo failed > ../logs/stage; exit 1; }
echo done > ../logs/stage
touch ../out/done

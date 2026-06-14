#!/usr/bin/env bash
# Concurrency sweep of ONE already-running engine: for each N, run the N-way
# SimulEval streaming eval and inline-score BLEU/StreamLAAL/StreamLAAL_CA.
# Writes per-N run_summary.json and a collated results.tsv.
#
#   ENGINE=vllm BASE_URL=http://127.0.0.1:8200 \
#   OUT_ROOT=/mnt/taurus/data2/jiaxuanluo/rasst_eval/runs/vllm_sweep \
#   bash eval/streaming_sst/run_sweep.sh
set -uo pipefail

SPACY=/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
REPO_ROOT=/mnt/taurus/home/jiaxuanluo/rasst-demo
ENGINE="${ENGINE:?set ENGINE=vllm|sglang}"
BASE_URL="${BASE_URL:?set BASE_URL}"
MODEL_NAME="${MODEL_NAME:-qwen3-omni}"
DATA_DIR="${DATA_DIR:-/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments}"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT}"
NLIST="${NLIST:-1 8 16 32}"
SEG_MS="${SEG_MS:-1920}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-40}"
# Optional: cap segments per run (throughput-only smoke; scoring is skipped when
# set, see run_concurrency.py --limit). 0 = full set (full BLEU/StreamLAAL).
LIMIT="${LIMIT:-0}"
LIMIT_ARG=()
[ "${LIMIT}" -gt 0 ] 2>/dev/null && LIMIT_ARG=(--limit "${LIMIT}")

mkdir -p "${OUT_ROOT}"
RES="${OUT_ROOT}/results.tsv"
printf "engine\tN\tsegments\twall_s\tseg_per_s\tBLEU\tStreamLAAL\tStreamLAAL_CA\trc_nonzero\n" > "${RES}"

cd "${REPO_ROOT}"
for N in ${NLIST}; do
  OUT="${OUT_ROOT}/${ENGINE}_n${N}"
  echo "[sweep] ===== ${ENGINE} N=${N} -> ${OUT} ====="
  "${SPACY}" eval/streaming_sst/run_concurrency.py \
    --engine "${ENGINE}" --base-url "${BASE_URL}" --model-name "${MODEL_NAME}" \
    --concurrency "${N}" --data-dir "${DATA_DIR}" --out-dir "${OUT}" \
    --source-segment-size "${SEG_MS}" --max-new-tokens "${MAX_NEW_TOKENS}" \
    ${LIMIT_ARG[@]+"${LIMIT_ARG[@]}"} --score < /dev/null
  # collate from run_summary.json
  "${SPACY}" - "${OUT}/run_summary.json" "${ENGINE}" "${N}" >> "${RES}" <<'PY'
import json, sys
s = json.load(open(sys.argv[1])); eng, N = sys.argv[2], sys.argv[3]
rc = sum(1 for r in s.get("worker_returncodes", []) if r != 0)
def g(k): 
    v = s.get(k); 
    return "" if v is None else (f"{v:.3f}" if isinstance(v,float) else str(v))
print("\t".join([eng, N, str(s.get("segments_total","")), g("wall_clock_sec"),
                 g("segments_per_sec"), g("BLEU"), g("StreamLAAL"), g("StreamLAAL_CA"), str(rc)]))
PY
done

echo "[sweep] DONE. results:"
cat "${RES}"

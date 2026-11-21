#!/usr/bin/env bash
# Serve AutoTerm-SST as evaluated in the demo paper: the RASST speech LLM with the
# language-model router (Qwen3-4B letter probabilities, sticky HMM filter, rho = 1e-3),
# at most K = 4 active slices within B = 4,000 terms, a 4-chunk router context, and
# routing updates every 5 seconds. The web UI is served on the same port.
#
# Required environment:
#   MODEL          RASST speech LLM (local path or Hugging Face id)
#   RETRIEVER      RASST MaxSim retriever checkpoint (rasst-hn1024.pt)
#   RETRIEVER_CODE directory that holds the RASST retriever package
#   MANIFEST       topic-slice manifest with the ten 1,000-term slices
# Optional environment:
#   PORT (8000), HOST (127.0.0.1), TP_SIZE (2), GPU_UTIL (0.80), MAX_SEQS (8),
#   ROUTER_LM (Qwen/Qwen3-4B-Instruct-2507), CONTEXT_MODEL (BAAI/bge-m3),
#   PROTOTYPES (configs/router_prototypes_generated_s0.json), PYTHON_BIN (python3),
#   DEFAULT_SLICE (nlp_core_1k), SLICES (the ten slice ids of the manifest)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL:?set MODEL to the RASST speech LLM}"
: "${RETRIEVER:?set RETRIEVER to the RASST retriever checkpoint}"
: "${RETRIEVER_CODE:?set RETRIEVER_CODE to the RASST retriever code directory}"
: "${MANIFEST:?set MANIFEST to the topic-slice manifest}"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
TP_SIZE="${TP_SIZE:-2}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_SEQS="${MAX_SEQS:-8}"
ROUTER_LM="${ROUTER_LM:-Qwen/Qwen3-4B-Instruct-2507}"
CONTEXT_MODEL="${CONTEXT_MODEL:-BAAI/bge-m3}"
PROTOTYPES="${PROTOTYPES:-${REPO_ROOT}/configs/router_prototypes_generated_s0.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_SLICE="${DEFAULT_SLICE:-nlp_core_1k}"
SLICES="${SLICES:-nlp_core_1k,medicine_core_1k,education_core_1k,finance_core_1k,legal_core_1k,environment_core_1k,entertainment_core_1k,science_core_1k,sports_core_1k,art_core_1k}"
TMP_DIR="${TMP_DIR:-${TMPDIR:-/tmp}/autoterm-sst-server-${PORT}}"
mkdir -p "${TMP_DIR}"

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -u eval/streaming_sst/serve_glossary_capacity_sweep.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --model-path "${MODEL}" \
  --manifest "${MANIFEST}" \
  --rag-model-path "${RETRIEVER}" \
  --rag-device cuda:0 \
  --required-presets "${SLICES}" \
  --autoterm-policy budgeted_top_slices \
  --autoterm-default-preset "${DEFAULT_SLICE}" \
  --autoterm-term-budget 4000 \
  --autoterm-max-active-slices 4 \
  --autoterm-context-window-chunks 4 \
  --autoterm-context-model "${CONTEXT_MODEL}" \
  --autoterm-context-device cpu \
  --autoterm-context-weight 1.0 \
  --autoterm-text-weight 0.0 \
  --autoterm-domain-probe-weight 0.0 \
  --autoterm-speech-weight 0.0 \
  --autoterm-metadata-weight 0.0 \
  --autoterm-ema-alpha 0.80 \
  --autoterm-update-sec 5.0 \
  --autoterm-warmup-sec 6.0 \
  --autoterm-cooldown-sec 30.0 \
  --autoterm-context-prototypes "${PROTOTYPES}" \
  --autoterm-context-scorer language_model \
  --autoterm-context-language-model "${ROUTER_LM}" \
  --autoterm-switch-rate 1e-3 \
  --no-autoterm-pin-active-slice \
  --vllm-tp-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --max-num-seqs "${MAX_SEQS}" \
  --max-model-len 16384 \
  --vllm-limit-audio 16 \
  --vllm-enforce-eager 1 \
  --enable-prefix-caching 1 \
  --disable-custom-all-reduce 1 \
  --vllm-use-v1 1 \
  --vllm-enable-v1-multiprocessing 1 \
  --vllm-worker-multiproc-method spawn \
  --vllm-moe-use-deep-gemm 0 \
  --vllm-use-fused-moe-grouped-topk 0 \
  --extra-python-path "${RETRIEVER_CODE}" \
  --scheduler-batch-size "${MAX_SEQS}" \
  --max-inflight-batches 2 \
  --max-new-tokens 40 \
  --rag-top-k 10 \
  --rag-score-threshold 0.78 \
  --retrieval-candidate-budget 100 \
  --term-map-format tagged \
  --empty-term-map-policy none_block \
  --tmp-dir "${TMP_DIR}" \
  --log-level info

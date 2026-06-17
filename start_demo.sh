#!/usr/bin/env bash
# Primary entry point: the thin RASST-Demo streaming SST framework.
#
# This replaces the three standalone servers as the single demo entry point.
# The framework (`framework.server`) is a thin transport/session/routing layer;
# it loads one or more agents and serves the existing `serve/static` UI unchanged.
#
#   agent_type "RASST"     -> framework.agents.omni.OmniAgent (in-process vLLM
#                             Qwen3-Omni, batched generate, optional MaxSim RAG)
#   agent_type "InfiniSST" -> framework.agents.infinisst.InfiniSSTAgent (scheduler)
#
# The old standalone servers remain available as legacy launchers:
#   scripts/legacy/start_infinisst_api.sh    (serve.api)
#   scripts/legacy/start_rasst_sglang.sh     (serve.rasst_sglang_server)
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/rasst-demo"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python}"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-/mnt/taurus/data2/jiaxuanluo/fairseq-0.12.2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export PYTHONPATH="${REPO_ROOT}:${FAIRSEQ_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export RASST_DEMO_MOCK="${RASST_DEMO_MOCK:-0}"

# Which agents the framework loads, and which one handles a blank/unknown
# agent_type. The UI's model picker uses these agent_type ids.
export RASST_FRAMEWORK_AGENTS="${RASST_FRAMEWORK_AGENTS:-InfiniSST,RASST}"
export RASST_FRAMEWORK_DEFAULT_AGENT="${RASST_FRAMEWORK_DEFAULT_AGENT:-RASST}"

# RASST/omni generation backend: in-process vLLM (batched generate -> 32+
# concurrent sessions/GPU). The model is loaded INTO this framework process, so
# run on a GPU host and set CUDA_VISIBLE_DEVICES (+ RASST_VLLM_TP_SIZE for TP).
export RASST_DEMO_LANGUAGE_PAIR="${RASST_DEMO_LANGUAGE_PAIR:-English -> Chinese}"
export RASST_VLLM_TP_SIZE="${RASST_VLLM_TP_SIZE:-1}"
export RASST_GPU_MEMORY_UTILIZATION="${RASST_GPU_MEMORY_UTILIZATION:-0.86}"
export RASST_MAX_NUM_SEQS="${RASST_MAX_NUM_SEQS:-32}"
export RASST_MAX_MODEL_LEN="${RASST_MAX_MODEL_LEN:-16384}"
export RASST_VLLM_LIMIT_AUDIO="${RASST_VLLM_LIMIT_AUDIO:-16}"
export RASST_ENABLE_PREFIX_CACHING="${RASST_ENABLE_PREFIX_CACHING:-1}"
export RASST_VLLM_ENFORCE_EAGER="${RASST_VLLM_ENFORCE_EAGER:-0}"
# Per-language model path is resolved from the catalog; override with
# RASST_VLLM_MODEL_PATH (or RASST_MODEL_ZH_CAP16_DENOISE / _JA_ / _DE_).
# Optional alternative backend instead of in-process vLLM: set the RASST omni
# template backend_kind to 'sglang_http' and point this at a vllm/sglang server.
export RASST_SGLANG_BASE_URL="${RASST_SGLANG_BASE_URL:-http://127.0.0.1:8100}"

if [ "${RASST_DEMO_MOCK}" != "1" ]; then
  unset RASST_DEMO_FAKE_GPUS
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${RASST_DEMO_FAKE_GPUS:-}" ]; then
  export RASST_DEMO_FAKE_GPUS="0"
fi

cd "${REPO_ROOT}"

echo "Starting RASST-Demo thin framework (framework.server)"
echo "  host: ${HOST}"
echo "  port: ${PORT}"
echo "  python: ${PYTHON_BIN}"
echo "  agents: ${RASST_FRAMEWORK_AGENTS} (default: ${RASST_FRAMEWORK_DEFAULT_AGENT})"
echo "  language pair: ${RASST_DEMO_LANGUAGE_PAIR}"
echo "  mock mode: ${RASST_DEMO_MOCK}"
if [ "${RASST_DEMO_MOCK}" != "1" ]; then
  echo "  RASST backend: in-process vLLM"
  echo "    tp_size=${RASST_VLLM_TP_SIZE} gpu_mem=${RASST_GPU_MEMORY_UTILIZATION} max_num_seqs=${RASST_MAX_NUM_SEQS}"
  echo "    max_model_len=${RASST_MAX_MODEL_LEN} limit_audio=${RASST_VLLM_LIMIT_AUDIO} cuda=${CUDA_VISIBLE_DEVICES:-<unset>}"
fi

exec "${PYTHON_BIN}" -m framework.server --host "${HOST}" --port "${PORT}"

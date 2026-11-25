#!/usr/bin/env bash
# vLLM OpenAI-compatible server for *pure* Qwen3-Omni-30B-A3B (no-RAG), TP=2.
# Mirrors the engine knobs used by scripts/run_taurus_framework_vllm.sh so the
# A/B vs sglang-omni differs only in the engine.
#
#   GPUS=2,3 PORT=8200 bash eval/streaming_sst/servers/serve_vllm_qwen3omni.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/rasst-demo}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
# Pure (no-RAG) Qwen3-Omni-30B-A3B-Instruct, resolved HF snapshot on Taurus.
MODEL_PATH="${MODEL_PATH:-/mnt/taurus/data2/jiaxuanluo/.cache/huggingface/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}"
SERVED_NAME="${SERVED_NAME:-qwen3-omni}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8200}"
GPUS="${GPUS:-2,3}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
LIMIT_AUDIO="${LIMIT_AUDIO:-16}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HOME="${HF_HOME:-/mnt/taurus/data2/jiaxuanluo/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/taurus/data2/jiaxuanluo/.cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
# vLLM 0.13 V1 + multiprocess TP knobs (same as the framework vLLM launcher).
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"

echo "[vllm] model=${MODEL_PATH} served=${SERVED_NAME} gpus=${GPUS} tp=${TP_SIZE} port=${PORT}"
nvidia-smi -L || true

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" --port "${PORT}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --limit-mm-per-prompt '{"audio": '"${LIMIT_AUDIO}"'}' \
  --enforce-eager \
  --disable-custom-all-reduce \
  --trust-remote-code

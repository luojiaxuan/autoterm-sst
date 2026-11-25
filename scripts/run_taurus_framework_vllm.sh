#!/usr/bin/env bash
# Resident (non-mock) real run of the thin framework on Taurus with the RASST
# Qwen3-Omni agent served by IN-PROCESS vLLM (tensor-parallel) + MaxSim RAG.
#
# Launch method: this node's SLURM view does not match physical GPU usage
# (GPUs 0-3 are busy with non-SLURM procs while SLURM reports the node idle),
# so we pin specific FREE GPUs via CUDA_VISIBLE_DEVICES and run resident with
# `setsid nohup` rather than sbatch. Example:
#
#   setsid nohup bash scripts/run_taurus_framework_vllm.sh \
#     > logs/framework_vllm_live.out 2>&1 &
#
# GPU layout (override CUDA_VISIBLE_DEVICES to change):
#   - vLLM uses the first RASST_VLLM_TP_SIZE visible GPUs (logical cuda:0..)
#   - RAG retriever uses RASST_RAG_DEVICE (a dedicated trailing GPU here)
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/rasst-demo}"
# spaCyEnv has vLLM 0.13.0 (native Qwen3-Omni multimodal) + torch cu128.
# The infinisst env's vLLM 0.9.2 only loads this checkpoint as a TEXT-only
# TransformersForCausalLM (rejects audio), so spaCyEnv is required here.
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8011}"

# Free GPUs on taurus: 5,6,7. vLLM tp=2 -> physical 6,7 ; RAG dedicated -> 5.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7,5}"

# serve/vllm_compat holds a sitecustomize.py that neutralizes vLLM 0.9.x's
# duplicate `aimv2` AutoConfig.register. It MUST be on PYTHONPATH (not just an
# in-process monkeypatch) so it also applies in vLLM's architecture-inspection
# subprocess (`python -m vllm.model_executor.models.registry`).
export PYTHONPATH="${REPO_ROOT}/serve/vllm_compat:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

# vLLM / NCCL knobs for this Omni MoE under tensor parallelism. vLLM 0.13 is
# V1-only, so we use V1 with multiprocessing (required for tp>1). NOTE: our
# VLLMBackend defaults VLLM_USE_V1=0 (older-vLLM mirror); we override to 1 here.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"

# Real run: no mock, RASST agent only (InfiniSST fairseq deps are absent here).
export RASST_DEMO_MOCK=0
export RASST_FRAMEWORK_AGENTS="${RASST_FRAMEWORK_AGENTS:-RASST}"
export RASST_FRAMEWORK_DEFAULT_AGENT="${RASST_FRAMEWORK_DEFAULT_AGENT:-RASST}"
export RASST_DEMO_LANGUAGE_PAIR="${RASST_DEMO_LANGUAGE_PAIR:-English -> Chinese}"

# In-process vLLM engine config.
export RASST_VLLM_TP_SIZE="${RASST_VLLM_TP_SIZE:-2}"
export RASST_GPU_MEMORY_UTILIZATION="${RASST_GPU_MEMORY_UTILIZATION:-0.80}"
export RASST_MAX_NUM_SEQS="${RASST_MAX_NUM_SEQS:-32}"
export RASST_SCHEDULER_BATCH_SIZE="${RASST_SCHEDULER_BATCH_SIZE:-32}"
export RASST_MAX_MODEL_LEN="${RASST_MAX_MODEL_LEN:-16384}"
export RASST_VLLM_LIMIT_AUDIO="${RASST_VLLM_LIMIT_AUDIO:-16}"
export RASST_VLLM_ENFORCE_EAGER="${RASST_VLLM_ENFORCE_EAGER:-1}"
export RASST_DISABLE_CUSTOM_ALL_REDUCE="${RASST_DISABLE_CUSTOM_ALL_REDUCE:-1}"
export RASST_ENABLE_PREFIX_CACHING="${RASST_ENABLE_PREFIX_CACHING:-1}"
export RASST_MAX_CACHE_CHUNKS="${RASST_MAX_CACHE_CHUNKS:-16}"
export RASST_KEEP_CACHE_CHUNKS="${RASST_KEEP_CACHE_CHUNKS:-8}"
export RASST_MAX_NEW_TOKENS="${RASST_MAX_NEW_TOKENS:-40}"
export RASST_TERM_MAP_FORMAT="${RASST_TERM_MAP_FORMAT:-tagged}"
export RASST_EMPTY_TERM_MAP_POLICY="${RASST_EMPTY_TERM_MAP_POLICY:-none_block}"

# RAG (MaxSim) retriever on a dedicated GPU (logical cuda:2 -> physical 5).
export RASST_ROOT="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_RAG_ENABLED="${RASST_RAG_ENABLED:-1}"
export RASST_RAG_DEVICE="${RASST_RAG_DEVICE:-cuda:2}"
export RASST_HN1024_RETRIEVER="${RASST_HN1024_RETRIEVER:-${REPO_ROOT}/checkpoints/retriever/rasst-hn1024.pt}"

cd "${REPO_ROOT}"
mkdir -p logs

echo "[INFO] host=$(hostname) port=${PORT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} tp=${RASST_VLLM_TP_SIZE} rag_device=${RASST_RAG_DEVICE}"
echo "[INFO] gpu_mem=${RASST_GPU_MEMORY_UTILIZATION} max_num_seqs=${RASST_MAX_NUM_SEQS} enforce_eager=${RASST_VLLM_ENFORCE_EAGER}"
echo "[INFO] agents=${RASST_FRAMEWORK_AGENTS} mock=${RASST_DEMO_MOCK} python=${PYTHON_BIN}"
nvidia-smi -L || true

exec "${PYTHON_BIN}" -m framework.server --host "${HOST}" --port "${PORT}"

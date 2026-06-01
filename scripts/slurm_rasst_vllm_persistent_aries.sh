#!/usr/bin/env bash
#SBATCH --job-name=rasst-vllm-live
#SBATCH --partition=aries
#SBATCH --nodelist=aries
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_vllm_live_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_vllm_live_%j.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/rasst-demo"
PORT="${PORT:-8010}"
PYTHON="${PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"

cd "${REPO_ROOT}"
mkdir -p logs

export RASST_ROOT="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ACTIVE_CODE_ROOT="${RASST_ACTIVE_CODE_ROOT:-${RASST_ROOT}/code/rasst}"
export PYTHONPATH="${REPO_ROOT}/serve/vllm_compat:${RASST_ACTIVE_CODE_ROOT}/eval:${RASST_ACTIVE_CODE_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_USE_FUSED_MOE_GROUPED_TOPK="${VLLM_USE_FUSED_MOE_GROUPED_TOPK:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE=1
export RASST_DEMO_LANGUAGE_PAIR="${RASST_DEMO_LANGUAGE_PAIR:-English -> Chinese}"
export RASST_HN1024_RETRIEVER="${RASST_HN1024_RETRIEVER:-${REPO_ROOT}/checkpoints/retriever/rasst-hn1024.pt}"
export RASST_RAG_ENABLED="${RASST_RAG_ENABLED:-1}"
export RASST_RAG_DEVICE="${RASST_RAG_DEVICE:-cuda:1}"
export RASST_VLLM_TP_SIZE="${RASST_VLLM_TP_SIZE:-2}"
export RASST_MAX_NUM_SEQS="${RASST_MAX_NUM_SEQS:-32}"
export RASST_SCHEDULER_BATCH_SIZE="${RASST_SCHEDULER_BATCH_SIZE:-32}"
export RASST_MAX_MODEL_LEN="${RASST_MAX_MODEL_LEN:-16384}"
export RASST_VLLM_LIMIT_AUDIO="${RASST_VLLM_LIMIT_AUDIO:-16}"
export RASST_MAX_CACHE_CHUNKS="${RASST_MAX_CACHE_CHUNKS:-16}"
export RASST_KEEP_CACHE_CHUNKS="${RASST_KEEP_CACHE_CHUNKS:-8}"
export RASST_MAX_NEW_TOKENS="${RASST_MAX_NEW_TOKENS:-40}"
export RASST_GPU_MEMORY_UTILIZATION="${RASST_GPU_MEMORY_UTILIZATION:-0.72}"
export RASST_BATCH_TIMEOUT="${RASST_BATCH_TIMEOUT:-0.05}"
export RASST_WORKER_GPUS="${RASST_WORKER_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] RASST_WORKER_GPUS=${RASST_WORKER_GPUS}"
echo "[INFO] RASST_VLLM_TP_SIZE=${RASST_VLLM_TP_SIZE}"
echo "[INFO] RASST_RAG_DEVICE=${RASST_RAG_DEVICE}"
echo "[INFO] port=${PORT}"
nvidia-smi -L || true

exec "${PYTHON}" -m serve.rasst_server --host 127.0.0.1 --port "${PORT}"

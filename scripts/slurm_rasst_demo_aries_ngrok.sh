#!/usr/bin/env bash
#SBATCH --job-name=rasst-vllm-demo
#SBATCH --partition=aries
#SBATCH --nodelist=aries
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_vllm_demo_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_vllm_demo_%j.err

set -euo pipefail

cd /mnt/taurus/home/jiaxuanluo/rasst-demo
mkdir -p logs

PORT="${PORT:-8000}"
NGROK_DOMAIN="${NGROK_DOMAIN:-amused-fleet-aardvark.ngrok-free.app}"
NGROK_BIN="${NGROK_BIN:-/mnt/aries/data6/jiaxuanluo/bin/ngrok}"
NGROK_CONFIG="${NGROK_CONFIG:-/mnt/taurus/home/jiaxuanluo/ngrok_jiaxuan.yml}"
PYTHON="${PYTHON:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python}"

export RASST_ROOT="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ACTIVE_CODE_ROOT="${RASST_ACTIVE_CODE_ROOT:-${RASST_ROOT}/code/rasst}"
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/rasst-demo/serve/vllm_compat:${RASST_ACTIVE_CODE_ROOT}/eval:${RASST_ACTIVE_CODE_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONNOUSERSITE=1
export RASST_DEMO_LANGUAGE_PAIR="${RASST_DEMO_LANGUAGE_PAIR:-English -> Chinese}"
export RASST_HN1024_RETRIEVER="${RASST_HN1024_RETRIEVER:-/mnt/taurus/home/jiaxuanluo/rasst-demo/checkpoints/retriever/rasst-hn1024.pt}"
export RASST_RAG_ENABLED="${RASST_RAG_ENABLED:-1}"
export RASST_MAX_NUM_SEQS="${RASST_MAX_NUM_SEQS:-16}"
export RASST_SCHEDULER_BATCH_SIZE="${RASST_SCHEDULER_BATCH_SIZE:-16}"
export RASST_MAX_MODEL_LEN="${RASST_MAX_MODEL_LEN:-16384}"
export RASST_VLLM_LIMIT_AUDIO="${RASST_VLLM_LIMIT_AUDIO:-16}"
export RASST_MAX_CACHE_CHUNKS="${RASST_MAX_CACHE_CHUNKS:-16}"
export RASST_KEEP_CACHE_CHUNKS="${RASST_KEEP_CACHE_CHUNKS:-8}"
export RASST_MAX_NEW_TOKENS="${RASST_MAX_NEW_TOKENS:-40}"
export RASST_GPU_MEMORY_UTILIZATION="${RASST_GPU_MEMORY_UTILIZATION:-0.86}"
export RASST_BATCH_TIMEOUT="${RASST_BATCH_TIMEOUT:-0.05}"
export RASST_WORKER_GPUS="${RASST_WORKER_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1}}"

SERVER_LOG="logs/rasst_vllm_server_${SLURM_JOB_ID}.log"
SERVER_ERR="logs/rasst_vllm_server_${SLURM_JOB_ID}.err"
NGROK_LOG="logs/rasst_vllm_ngrok_${SLURM_JOB_ID}.log"

cleanup() {
  if [ -n "${NGROK_PID:-}" ]; then
    kill "${NGROK_PID}" 2>/dev/null || true
    wait "${NGROK_PID}" 2>/dev/null || true
  fi
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[INFO] job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] RASST_WORKER_GPUS=${RASST_WORKER_GPUS}"
echo "[INFO] language_pair=${RASST_DEMO_LANGUAGE_PAIR}"
echo "[INFO] starting RASST server on 127.0.0.1:${PORT}"

HOST=127.0.0.1 PORT="${PORT}" "${PYTHON}" -m serve.rasst_server \
  --host 127.0.0.1 \
  --port "${PORT}" \
  >"${SERVER_LOG}" 2>"${SERVER_ERR}" &
SERVER_PID=$!

for i in $(seq 1 1800); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] RASST server exited before becoming healthy"
    tail -160 "${SERVER_LOG}" || true
    tail -160 "${SERVER_ERR}" || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/rasst_vllm_health_${SLURM_JOB_ID}.json 2>/dev/null; then
    if grep -q '"status":"healthy"' /tmp/rasst_vllm_health_${SLURM_JOB_ID}.json; then
      echo "[INFO] RASST server is healthy after ${i}s"
      cat /tmp/rasst_vllm_health_${SLURM_JOB_ID}.json
      break
    fi
    if [ $((i % 30)) -eq 0 ]; then
      echo "[INFO] still loading after ${i}s"
      cat /tmp/rasst_vllm_health_${SLURM_JOB_ID}.json || true
    fi
  fi
  sleep 1
done

if ! grep -q '"status":"healthy"' /tmp/rasst_vllm_health_${SLURM_JOB_ID}.json 2>/dev/null; then
  echo "[ERROR] RASST server did not become healthy"
  tail -160 "${SERVER_LOG}" || true
  tail -160 "${SERVER_ERR}" || true
  exit 1
fi

echo "[INFO] starting ngrok tunnel https://${NGROK_DOMAIN} -> http://127.0.0.1:${PORT}"
"${NGROK_BIN}" http --url="${NGROK_DOMAIN}" --config "${NGROK_CONFIG}" "${PORT}" >"${NGROK_LOG}" 2>&1 &
NGROK_PID=$!

for i in $(seq 1 120); do
  if curl -k -fsS "https://${NGROK_DOMAIN}/health" >/tmp/rasst_vllm_ngrok_health_${SLURM_JOB_ID}.json 2>/dev/null; then
    echo "[INFO] ngrok health is reachable after ${i}s"
    cat /tmp/rasst_vllm_ngrok_health_${SLURM_JOB_ID}.json
    echo "[INFO] demo URL: https://${NGROK_DOMAIN}/"
    wait "${NGROK_PID}"
    exit $?
  fi
  if ! kill -0 "${NGROK_PID}" 2>/dev/null; then
    echo "[ERROR] ngrok exited before remote health became reachable"
    tail -120 "${NGROK_LOG}" || true
    exit 1
  fi
  sleep 1
done

echo "[ERROR] ngrok tunnel did not become reachable"
tail -120 "${NGROK_LOG}" || true
exit 1

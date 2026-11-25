#!/usr/bin/env bash
#SBATCH --job-name=rasst-real-demo
#SBATCH --partition=aries
#SBATCH --nodelist=aries
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_real_demo_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_real_demo_%j.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/rasst-demo"
PORT="${PORT:-8000}"
NGROK_DOMAIN="${NGROK_DOMAIN:-amused-fleet-aardvark.ngrok-free.app}"
NGROK_BIN="${NGROK_BIN:-/mnt/aries/data6/jiaxuanluo/bin/ngrok}"
NGROK_CONFIG="${NGROK_CONFIG:-/mnt/taurus/home/jiaxuanluo/ngrok_jiaxuan.yml}"

cd "${REPO_ROOT}"
mkdir -p logs

echo "[INFO] host=$(hostname)"
echo "[INFO] job_id=${SLURM_JOB_ID:-none}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

unset RASST_DEMO_FAKE_GPUS
export RASST_DEMO_MOCK=0
export RASST_DEMO_ALLOW_MOCK_ON_FAILURE=0
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

SERVER_LOG="logs/real_demo_server_${SLURM_JOB_ID}.log"
SERVER_ERR="logs/real_demo_server_${SLURM_JOB_ID}.err"
NGROK_LOG="logs/real_demo_ngrok_${SLURM_JOB_ID}.log"

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

HOST=127.0.0.1 PORT="${PORT}" ./start_demo.sh >"${SERVER_LOG}" 2>"${SERVER_ERR}" &
SERVER_PID=$!

echo "[INFO] server_pid=${SERVER_PID}"
echo "[INFO] waiting for server health on http://127.0.0.1:${PORT}/health"

for i in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/rasst_demo_health_${SLURM_JOB_ID}.json 2>/dev/null; then
    echo "[INFO] server health is reachable after ${i}s"
    cat /tmp/rasst_demo_health_${SLURM_JOB_ID}.json
    echo
    break
  fi

  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[ERROR] server exited before health became reachable"
    echo "[ERROR] server stdout tail:"
    tail -80 "${SERVER_LOG}" || true
    echo "[ERROR] server stderr tail:"
    tail -80 "${SERVER_ERR}" || true
    exit 1
  fi

  sleep 1
done

if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/rasst_demo_health_${SLURM_JOB_ID}.json; then
  echo "[ERROR] server did not become healthy"
  tail -120 "${SERVER_LOG}" || true
  tail -120 "${SERVER_ERR}" || true
  exit 1
fi

if grep -q '"mock_mode":true' /tmp/rasst_demo_health_${SLURM_JOB_ID}.json; then
  echo "[ERROR] server is in mock mode; refusing to expose demo"
  cat /tmp/rasst_demo_health_${SLURM_JOB_ID}.json
  exit 1
fi

echo "[INFO] starting ngrok tunnel https://${NGROK_DOMAIN} -> http://127.0.0.1:${PORT}"
"${NGROK_BIN}" http --url="${NGROK_DOMAIN}" --config "${NGROK_CONFIG}" "${PORT}" >"${NGROK_LOG}" 2>&1 &
NGROK_PID=$!

for i in $(seq 1 60); do
  if curl -k -fsS "https://${NGROK_DOMAIN}/health" >/tmp/rasst_demo_ngrok_health_${SLURM_JOB_ID}.json 2>/dev/null; then
    echo "[INFO] ngrok health is reachable after ${i}s"
    cat /tmp/rasst_demo_ngrok_health_${SLURM_JOB_ID}.json
    echo
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

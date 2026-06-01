#!/usr/bin/env bash
#SBATCH --job-name=rasst-real-p0
#SBATCH --partition=aries
#SBATCH --nodelist=aries
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_real_p0_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_real_p0_%j.err

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/rasst-demo"
PYTHON_BIN="/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python"
PORT="${PORT:-18000}"

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

SERVER_LOG="logs/real_server_${SLURM_JOB_ID}.log"
SERVER_ERR="logs/real_server_${SLURM_JOB_ID}.err"

HOST=127.0.0.1 PORT="${PORT}" ./start_demo.sh >"${SERVER_LOG}" 2>"${SERVER_ERR}" &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true; wait ${SERVER_PID} 2>/dev/null || true' EXIT

echo "[INFO] server_pid=${SERVER_PID}"
echo "[INFO] waiting for server health"

for i in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/rasst_health_${SLURM_JOB_ID}.json 2>/dev/null; then
    echo "[INFO] server health is reachable after ${i}s"
    cat /tmp/rasst_health_${SLURM_JOB_ID}.json
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

if ! curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/rasst_health_${SLURM_JOB_ID}.json; then
  echo "[ERROR] server did not become healthy"
  tail -120 "${SERVER_LOG}" || true
  tail -120 "${SERVER_ERR}" || true
  exit 1
fi

if grep -q '"mock_mode":true' /tmp/rasst_health_${SLURM_JOB_ID}.json; then
  echo "[ERROR] server is in mock mode; refusing to run real test"
  cat /tmp/rasst_health_${SLURM_JOB_ID}.json
  exit 1
fi

echo "[INFO] running one-session real protocol smoke"
"${PYTHON_BIN}" scripts/smoke_p0_protocol.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --sessions 1 \
  --samples 16000 \
  --timeout 240

echo "[INFO] running 32-session real protocol smoke"
"${PYTHON_BIN}" scripts/smoke_p0_protocol.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --sessions 32 \
  --samples 16000 \
  --timeout 300

echo "[INFO] final health"
curl -fsS "http://127.0.0.1:${PORT}/health"
echo

echo "[INFO] server stdout tail:"
tail -120 "${SERVER_LOG}" || true
echo "[INFO] server stderr tail:"
tail -120 "${SERVER_ERR}" || true

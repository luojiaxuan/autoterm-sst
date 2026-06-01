#!/usr/bin/env bash
#SBATCH --job-name=rasst-demo-live
#SBATCH --partition=aries
#SBATCH --nodelist=aries
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=360G
#SBATCH --time=24:00:00
#SBATCH --chdir=/mnt/taurus/home/jiaxuanluo/rasst-demo
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/persistent_3gpu_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/persistent_3gpu_%j.err

set -euo pipefail

cd /mnt/taurus/home/jiaxuanluo/rasst-demo
mkdir -p logs

IMAGE="${IMAGE:-frankleeeee/sglang-omni:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-rasst_sglang_live_${SLURM_JOB_ID}}"
RASST_PORT="${RASST_PORT:-8000}"
RASST_SGLANG_PORT="${RASST_SGLANG_PORT:-8100}"
INFINISST_PORT="${INFINISST_PORT:-8001}"
LANGUAGE_PAIR="${RASST_DEMO_LANGUAGE_PAIR:-English -> Chinese}"

export RASST_ROOT="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ACTIVE_CODE_ROOT="${RASST_ACTIVE_CODE_ROOT:-${RASST_ROOT}/code/rasst}"
export RASST_HN1024_RETRIEVER="${RASST_HN1024_RETRIEVER:-/mnt/taurus/home/jiaxuanluo/rasst-demo/checkpoints/retriever/rasst-hn1024.pt}"
export RASST_DEMO_DATA_ROOT="${RASST_DEMO_DATA_ROOT:-/mnt/taurus/data2/jiaxuanluo/rasst-demo}"
export RASST_DEMO_LANGUAGE_PAIR="${LANGUAGE_PAIR}"
export RASST_RAG_ENABLED="${RASST_RAG_ENABLED:-1}"
export RASST_RAG_PROFILE="${RASST_RAG_PROFILE:-1}"
export RASST_SCHEDULER_BATCH_SIZE="${RASST_SCHEDULER_BATCH_SIZE:-32}"
export RASST_MAX_NEW_TOKENS="${RASST_MAX_NEW_TOKENS:-40}"
export RASST_KEEP_CACHE_CHUNKS="${RASST_KEEP_CACHE_CHUNKS:-8}"
export RASST_SGLANG_MEM_FRACTION_STATIC="${RASST_SGLANG_MEM_FRACTION_STATIC:-0.75}"
export SGLANG_OMNI_STARTUP_TIMEOUT="${SGLANG_OMNI_STARTUP_TIMEOUT:-1800}"
export RASST_TMP_DIR="${RASST_TMP_DIR:-/dev/shm/rasst_sglang_live_${SLURM_JOB_ID}}"
export RASST_SGLANG_BASE_URL="http://127.0.0.1:${RASST_SGLANG_PORT}"
export INFINISST_BASE_URL="http://127.0.0.1:${INFINISST_PORT}"

case "${LANGUAGE_PAIR}" in
  "English -> Chinese")
    MODEL_PATH="${RASST_MODEL_ZH_CAP16_DENOISE:-/mnt/taurus/data2/jiaxuanluo/RASST_release_runs/models/speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf}"
    ;;
  "English -> Japanese")
    MODEL_PATH="${RASST_MODEL_JA_CAP16_DENOISE:-/mnt/taurus/data1/jiaxuanluo/slm_local_cache/ja_tagged_acl_20260525/cap16_denoise_ttag/v2-20260525-235251-hf}"
    ;;
  "English -> German")
    MODEL_PATH="${RASST_MODEL_DE_CAP16_DENOISE:-/mnt/taurus/data1/jiaxuanluo/slm_local_cache/de_tagged_acl_20260525/cap16_denoise_ttag/v0-20260525-203735-hf}"
    ;;
  *)
    echo "[ERROR] unsupported RASST_DEMO_LANGUAGE_PAIR=${LANGUAGE_PAIR}" >&2
    exit 2
    ;;
esac
export RASST_SGLANG_MODEL_PATH="${MODEL_PATH}"

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
if [ "${CUDA_VISIBLE_DEVICES:-}" = "all" ]; then
  ALLOCATED_GPUS=(0 1 2)
fi
if [ "${#ALLOCATED_GPUS[@]}" -lt 3 ]; then
  echo "[ERROR] expected 3 allocated GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi
RASST_GPU_SPEC="${ALLOCATED_GPUS[0]},${ALLOCATED_GPUS[1]}"
INFINISST_GPU_SPEC="${ALLOCATED_GPUS[2]}"
RASST_CONTAINER_GPU_SPEC="${RASST_GPU_SPEC}"
if [ "${RASST_RAG_ON_INFINISST:-0}" = "1" ]; then
  export RASST_RAG_DEVICE="cuda:${INFINISST_GPU_SPEC}"
  RASST_CONTAINER_GPU_SPEC="${RASST_GPU_SPEC},${INFINISST_GPU_SPEC}"
else
  export RASST_RAG_DEVICE="${RASST_RAG_DEVICE:-cuda:${ALLOCATED_GPUS[1]}}"
fi

RASST_CONTAINER_LOG="logs/persistent_rasst_container_${SLURM_JOB_ID}.log"
RASST_CONTAINER_ERR="logs/persistent_rasst_container_${SLURM_JOB_ID}.err"
INFINISST_LOG="logs/persistent_infinisst_${SLURM_JOB_ID}.log"
INFINISST_ERR="logs/persistent_infinisst_${SLURM_JOB_ID}.err"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  if [ -n "${INFINISST_PID:-}" ]; then
    kill "${INFINISST_PID}" >/dev/null 2>&1 || true
    wait "${INFINISST_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[INFO] job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] RASST SGLang TP GPUs=${RASST_GPU_SPEC}; RASST container GPUs=${RASST_CONTAINER_GPU_SPEC}; RAG device=${RASST_RAG_DEVICE}; InfiniSST host GPU=${INFINISST_GPU_SPEC}"
echo "[INFO] RASST UI/proxy=http://127.0.0.1:${RASST_PORT}"
echo "[INFO] InfiniSST direct=http://127.0.0.1:${INFINISST_PORT}"

wait_for_port_free() {
  local port="$1"
  local name="$2"
  for i in $(seq 1 60); do
    if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      return 0
    fi
    echo "[INFO] waiting for ${name} port ${port} to be released (${i}/60)"
    sleep 1
  done
  echo "[ERROR] ${name} port ${port} is still in use" >&2
  ss -ltnp 2>/dev/null | grep -E ":${port}\\b" >&2 || true
  exit 1
}

wait_for_port_free "${RASST_PORT}" "RASST wrapper"
wait_for_port_free "${RASST_SGLANG_PORT}" "RASST SGLang"
wait_for_port_free "${INFINISST_PORT}" "InfiniSST"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "[INFO] starting InfiniSST on host GPU ${INFINISST_GPU_SPEC}"
(
  export CUDA_VISIBLE_DEVICES="${INFINISST_GPU_SPEC}"
  unset RASST_DEMO_FAKE_GPUS
  export RASST_DEMO_MOCK=0
  export RASST_DEMO_ALLOW_MOCK_ON_FAILURE=0
  export PYTHONUNBUFFERED=1
  export PYTHONNOUSERSITE=1
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
  HOST=127.0.0.1 PORT="${INFINISST_PORT}" ./start_demo.sh
) >"${INFINISST_LOG}" 2>"${INFINISST_ERR}" &
INFINISST_PID=$!

echo "[INFO] starting RASST SGLang-Omni container on GPUs ${RASST_CONTAINER_GPU_SPEC}"
docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus "\"device=${RASST_CONTAINER_GPU_SPEC}\"" \
  --ipc host \
  --network host \
  --privileged \
  --shm-size 64g \
  -v /mnt:/mnt \
  -v /home:/home \
  -w /mnt/taurus/home/jiaxuanluo/rasst-demo \
  -e HF_HOME=/mnt/taurus/data2/jiaxuanluo/.cache/huggingface \
  -e XDG_CACHE_HOME=/mnt/taurus/data2/jiaxuanluo/.cache \
  -e TORCH_HOME=/mnt/taurus/data2/jiaxuanluo/.cache/torch \
  -e RASST_ROOT \
  -e RASST_ACTIVE_CODE_ROOT \
  -e RASST_HN1024_RETRIEVER \
  -e RASST_DEMO_DATA_ROOT \
  -e RASST_DEMO_LANGUAGE_PAIR \
  -e RASST_RAG_DEVICE \
  -e RASST_RAG_ENABLED \
  -e RASST_RAG_PROFILE \
  -e RASST_SCHEDULER_BATCH_SIZE \
  -e RASST_MAX_NEW_TOKENS \
  -e RASST_KEEP_CACHE_CHUNKS \
  -e RASST_SGLANG_MEM_FRACTION_STATIC \
  -e RASST_TMP_DIR \
  -e RASST_SGLANG_BASE_URL \
  -e RASST_SGLANG_MODEL_PATH \
  -e INFINISST_BASE_URL \
  -e SGLANG_OMNI_STARTUP_TIMEOUT \
  -e PYTHONUNBUFFERED=1 \
  -e TOKENIZERS_PARALLELISM=false \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail
cd /mnt/taurus/home/jiaxuanluo/rasst-demo
export PYTHONPATH="/mnt/taurus/home/jiaxuanluo/sglang-omni:${RASST_ACTIVE_CODE_ROOT}/eval:${RASST_ACTIVE_CODE_ROOT}:/mnt/taurus/data2/jiaxuanluo/RASST/code/legacy/documents/code/general:${PYTHONPATH:-}"
mkdir -p logs "${RASST_TMP_DIR}"

python - <<PY >/tmp/rasst_sglang_import_check.log 2>&1 || python -m pip install -q msgpack typer av qwen-vl-utils==0.0.11 librosa==0.11.0 numba==0.63.1 peft==0.13.2 websockets
import sglang_omni.serve  # noqa: F401
import typer  # noqa: F401
import msgpack  # noqa: F401
import av  # noqa: F401
import qwen_vl_utils  # noqa: F401
import librosa  # noqa: F401
import peft  # noqa: F401
import websockets  # noqa: F401
assert peft.__version__.startswith("0.13."), peft.__version__
PY

python scripts/sglang_omni_qwen3_text_tp_server.py \
  --model-path "${RASST_SGLANG_MODEL_PATH}" \
  --host 127.0.0.1 \
  --port '"${RASST_SGLANG_PORT}"' \
  --model-name rasst-qwen3-omni \
  --pipeline-name rasst-live \
  --ipc-base-path /tmp/rlive'"${SLURM_JOB_ID}"' \
  --thinker-tp-size 2 \
  --thinker-gpus '"${RASST_GPU_SPEC}"' \
  --encoder-gpu '"${ALLOCATED_GPUS[1]}"' \
  --thinker-max-seq-len "${RASST_SGLANG_MAX_SEQ_LEN:-8192}" \
  --max-running-requests "${RASST_SGLANG_MAX_RUNNING_REQUESTS:-32}" \
  --max-prefill-tokens "${RASST_SGLANG_MAX_PREFILL_TOKENS:-16384}" \
  --mem-fraction-static "${RASST_SGLANG_MEM_FRACTION_STATIC}" \
  >logs/persistent_sglang_engine_'"${SLURM_JOB_ID}"'.log \
  2>logs/persistent_sglang_engine_'"${SLURM_JOB_ID}"'.err &
SGLANG_PID=$!

cleanup_inner() {
  kill "${WRAPPER_PID:-0}" >/dev/null 2>&1 || true
  kill "${SGLANG_PID:-0}" >/dev/null 2>&1 || true
  wait "${WRAPPER_PID:-0}" >/dev/null 2>&1 || true
  wait "${SGLANG_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup_inner EXIT

for i in $(seq 1 2400); do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "[ERROR] SGLang server exited before health"
    tail -160 logs/persistent_sglang_engine_'"${SLURM_JOB_ID}"'.err || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:'"${RASST_SGLANG_PORT}"'/health" >/tmp/rasst_sglang_live_health_'"${SLURM_JOB_ID}"'.json 2>/dev/null; then
    if grep -q "\"status\":\"healthy\"" /tmp/rasst_sglang_live_health_'"${SLURM_JOB_ID}"'.json; then
      echo "[INFO] SGLang healthy after ${i}s"
      cat /tmp/rasst_sglang_live_health_'"${SLURM_JOB_ID}"'.json
      break
    fi
  fi
  sleep 1
done

python -m serve.rasst_sglang_server \
  --host 127.0.0.1 \
  --port '"${RASST_PORT}"' \
  >logs/persistent_rasst_wrapper_'"${SLURM_JOB_ID}"'.log \
  2>logs/persistent_rasst_wrapper_'"${SLURM_JOB_ID}"'.err &
WRAPPER_PID=$!

for i in $(seq 1 900); do
  if ! kill -0 "${WRAPPER_PID}" 2>/dev/null; then
    echo "[ERROR] RASST wrapper exited before health"
    tail -160 logs/persistent_rasst_wrapper_'"${SLURM_JOB_ID}"'.err || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:'"${RASST_PORT}"'/health" >/tmp/rasst_wrapper_live_health_'"${SLURM_JOB_ID}"'.json 2>/dev/null; then
    if grep -q "\"status\":\"healthy\"" /tmp/rasst_wrapper_live_health_'"${SLURM_JOB_ID}"'.json; then
      echo "[INFO] RASST wrapper healthy after ${i}s"
      cat /tmp/rasst_wrapper_live_health_'"${SLURM_JOB_ID}"'.json
      break
    fi
  fi
  sleep 1
done

echo "[INFO] RASST service is persistent"
while kill -0 "${SGLANG_PID}" 2>/dev/null && kill -0 "${WRAPPER_PID}" 2>/dev/null; do
  sleep 30
done
echo "[ERROR] RASST service process exited"
exit 1
' >"${RASST_CONTAINER_LOG}" 2>"${RASST_CONTAINER_ERR}" &
RASST_CONTAINER_PID=$!

wait_for_health() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local log="$4"
  local err="$5"
  for i in $(seq 1 2400); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "[ERROR] ${name} process exited before health"
      tail -120 "${log}" || true
      tail -120 "${err}" || true
      exit 1
    fi
    if curl -fsS "${url}/health" >/tmp/${name}_health_${SLURM_JOB_ID}.json 2>/dev/null; then
      echo "[INFO] ${name} healthy after ${i}s"
      cat /tmp/${name}_health_${SLURM_JOB_ID}.json
      echo
      return 0
    fi
    sleep 1
  done
  echo "[ERROR] ${name} did not become healthy"
  tail -120 "${log}" || true
  tail -120 "${err}" || true
  exit 1
}

wait_for_health "infinisst" "http://127.0.0.1:${INFINISST_PORT}" "${INFINISST_PID}" "${INFINISST_LOG}" "${INFINISST_ERR}"
wait_for_health "rasst" "http://127.0.0.1:${RASST_PORT}" "${RASST_CONTAINER_PID}" "${RASST_CONTAINER_LOG}" "${RASST_CONTAINER_ERR}"

if grep -q '"mock_mode":true' /tmp/infinisst_health_${SLURM_JOB_ID}.json; then
  echo "[ERROR] InfiniSST is in mock mode"
  exit 1
fi
if grep -q '"mock_mode":true' /tmp/rasst_health_${SLURM_JOB_ID}.json; then
  echo "[ERROR] RASST is in mock mode"
  exit 1
fi

echo "[INFO] persistent 3GPU demo ready"
echo "[INFO] RASST + proxy UI: http://127.0.0.1:${RASST_PORT}/"
echo "[INFO] InfiniSST direct: http://127.0.0.1:${INFINISST_PORT}/"
echo "[INFO] RASST container log: ${RASST_CONTAINER_LOG}"
echo "[INFO] InfiniSST log: ${INFINISST_LOG}"

while true; do
  if ! kill -0 "${RASST_CONTAINER_PID}" 2>/dev/null; then
    echo "[ERROR] RASST container exited"
    tail -120 "${RASST_CONTAINER_LOG}" || true
    tail -120 "${RASST_CONTAINER_ERR}" || true
    exit 1
  fi
  if ! kill -0 "${INFINISST_PID}" 2>/dev/null; then
    echo "[ERROR] InfiniSST server exited"
    tail -120 "${INFINISST_LOG}" || true
    tail -120 "${INFINISST_ERR}" || true
    exit 1
  fi
  sleep 30
done

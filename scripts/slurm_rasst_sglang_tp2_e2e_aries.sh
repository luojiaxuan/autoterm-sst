#!/usr/bin/env bash
#SBATCH --job-name=rasst-sglang
#SBATCH --partition=aries
#SBATCH --nodelist=aries
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=260G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_sglang_tp2_%j.log
#SBATCH --error=/mnt/taurus/home/jiaxuanluo/rasst-demo/logs/rasst_sglang_tp2_%j.err

set -euo pipefail

cd /mnt/taurus/home/jiaxuanluo/rasst-demo
mkdir -p logs

IMAGE="${IMAGE:-frankleeeee/sglang-omni:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-rasst_sglang_${SLURM_JOB_ID}}"
PORT="${PORT:-8000}"
SGLANG_PORT="${SGLANG_PORT:-8100}"
LANGUAGE_PAIR="${RASST_DEMO_LANGUAGE_PAIR:-English -> Chinese}"
RUN_FULL_STRESS="${RUN_FULL_STRESS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"

export RASST_ROOT="${RASST_ROOT:-/mnt/taurus/data2/jiaxuanluo/RASST}"
export RASST_ACTIVE_CODE_ROOT="${RASST_ACTIVE_CODE_ROOT:-${RASST_ROOT}/code/rasst}"
export RASST_HN1024_RETRIEVER="${RASST_HN1024_RETRIEVER:-/mnt/taurus/home/jiaxuanluo/rasst-demo/checkpoints/retriever/rasst-hn1024.pt}"
export RASST_DEMO_LANGUAGE_PAIR="${LANGUAGE_PAIR}"
export RASST_RAG_DEVICE="${RASST_RAG_DEVICE:-cuda:1}"
export RASST_RAG_ENABLED="${RASST_RAG_ENABLED:-1}"
export RASST_SCHEDULER_BATCH_SIZE="${RASST_SCHEDULER_BATCH_SIZE:-32}"
export RASST_MAX_NEW_TOKENS="${RASST_MAX_NEW_TOKENS:-40}"
export RASST_KEEP_CACHE_CHUNKS="${RASST_KEEP_CACHE_CHUNKS:-8}"
export RASST_SGLANG_MEM_FRACTION_STATIC="${RASST_SGLANG_MEM_FRACTION_STATIC:-0.75}"
export SGLANG_OMNI_STARTUP_TIMEOUT="${SGLANG_OMNI_STARTUP_TIMEOUT:-1800}"
export RASST_TMP_DIR="${RASST_TMP_DIR:-/dev/shm/rasst_sglang_${SLURM_JOB_ID}}"
export RASST_SGLANG_BASE_URL="http://127.0.0.1:${SGLANG_PORT}"

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

AUDIO_FILE="${AUDIO_FILE:-/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/audio/acl6060/2022.acl-long.590.wav}"

SERVER_LOG="logs/rasst_sglang_wrapper_${SLURM_JOB_ID}.log"
SERVER_ERR="logs/rasst_sglang_wrapper_${SLURM_JOB_ID}.err"
SGLANG_LOG="logs/rasst_sglang_engine_${SLURM_JOB_ID}.log"
SGLANG_ERR="logs/rasst_sglang_engine_${SLURM_JOB_ID}.err"
SMOKE_LOG="logs/rasst_sglang_smoke_${SLURM_JOB_ID}.log"
STRESS_LOG="logs/rasst_sglang_stress32_${SLURM_JOB_ID}.log"

cleanup_container() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup_container EXIT

echo "[INFO] job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[INFO] docker image=${IMAGE}"
echo "[INFO] language_pair=${LANGUAGE_PAIR}"
echo "[INFO] model_path=${MODEL_PATH}"
echo "[INFO] rag_device=${RASST_RAG_DEVICE}"
echo "[INFO] audio_file=${AUDIO_FILE}"

GPU_DEVICE_SPEC="${CUDA_VISIBLE_DEVICES:-all}"
if [[ "${GPU_DEVICE_SPEC}" == "all" || -z "${GPU_DEVICE_SPEC}" ]]; then
  DOCKER_GPU_ARGS=(--gpus all)
else
  # Docker requires the device request to keep the comma-separated ids quoted.
  DOCKER_GPU_ARGS=(--gpus "\"device=${GPU_DEVICE_SPEC}\"")
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --rm \
  --name "${CONTAINER_NAME}" \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc host \
  --network host \
  --privileged \
  --shm-size 64g \
  -v /mnt:/mnt \
  -v /home:/home \
  -w /mnt/taurus/home/jiaxuanluo/rasst-demo \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e HF_HOME=/mnt/taurus/data2/jiaxuanluo/.cache/huggingface \
  -e XDG_CACHE_HOME=/mnt/taurus/data2/jiaxuanluo/.cache \
  -e TORCH_HOME=/mnt/taurus/data2/jiaxuanluo/.cache/torch \
  -e RASST_ROOT \
  -e RASST_ACTIVE_CODE_ROOT \
  -e RASST_HN1024_RETRIEVER \
  -e RASST_DEMO_LANGUAGE_PAIR \
  -e RASST_RAG_DEVICE \
  -e RASST_RAG_ENABLED \
  -e RASST_SCHEDULER_BATCH_SIZE \
  -e RASST_MAX_NEW_TOKENS \
  -e RASST_KEEP_CACHE_CHUNKS \
  -e RASST_SGLANG_MEM_FRACTION_STATIC \
  -e RASST_TMP_DIR \
  -e RASST_SGLANG_BASE_URL \
  -e RASST_SGLANG_MODEL_PATH \
  -e SGLANG_OMNI_STARTUP_TIMEOUT \
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

echo "[CONTAINER] python=$(python --version 2>&1)"
echo "[CONTAINER] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
python - <<PY
import torch
print("[CONTAINER] torch", torch.__version__, "cuda", torch.version.cuda, "visible", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("[CONTAINER] gpu", i, torch.cuda.get_device_name(i))
PY

python scripts/sglang_omni_qwen3_text_tp_server.py \
  --model-path "${RASST_SGLANG_MODEL_PATH}" \
  --host 127.0.0.1 \
  --port '"${SGLANG_PORT}"' \
  --model-name rasst-qwen3-omni \
  --pipeline-name rasst \
  --ipc-base-path /tmp/r'"${SLURM_JOB_ID}"' \
  --thinker-tp-size 2 \
  --thinker-gpus 0,1 \
  --encoder-gpu 1 \
  --thinker-max-seq-len "${RASST_SGLANG_MAX_SEQ_LEN:-8192}" \
  --max-running-requests "${RASST_SGLANG_MAX_RUNNING_REQUESTS:-32}" \
  --max-prefill-tokens "${RASST_SGLANG_MAX_PREFILL_TOKENS:-16384}" \
  --mem-fraction-static "${RASST_SGLANG_MEM_FRACTION_STATIC}" \
  >"'"${SGLANG_LOG}"'" 2>"'"${SGLANG_ERR}"'" &
SGLANG_PID=$!

cleanup() {
  kill "${WRAPPER_PID:-0}" >/dev/null 2>&1 || true
  kill "${SGLANG_PID:-0}" >/dev/null 2>&1 || true
  wait "${WRAPPER_PID:-0}" >/dev/null 2>&1 || true
  wait "${SGLANG_PID:-0}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for i in $(seq 1 2400); do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "[ERROR] SGLang server exited before health"
    tail -160 "'"${SGLANG_LOG}"'" || true
    tail -160 "'"${SGLANG_ERR}"'" || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:'"${SGLANG_PORT}"'/health" >/tmp/rasst_sglang_health_'"${SLURM_JOB_ID}"'.json 2>/dev/null; then
    if grep -q "\"status\":\"healthy\"" /tmp/rasst_sglang_health_'"${SLURM_JOB_ID}"'.json; then
      echo "[INFO] SGLang healthy after ${i}s"
      cat /tmp/rasst_sglang_health_'"${SLURM_JOB_ID}"'.json
      break
    fi
  fi
  if [ $((i % 60)) -eq 0 ]; then
    echo "[INFO] SGLang still loading after ${i}s"
    tail -40 "'"${SGLANG_LOG}"'" || true
    tail -80 "'"${SGLANG_ERR}"'" || true
  fi
  sleep 1
done

if ! grep -q "\"status\":\"healthy\"" /tmp/rasst_sglang_health_'"${SLURM_JOB_ID}"'.json 2>/dev/null; then
  echo "[ERROR] SGLang did not become healthy"
  tail -160 "'"${SGLANG_LOG}"'" || true
  tail -160 "'"${SGLANG_ERR}"'" || true
  exit 1
fi

python -m serve.rasst_sglang_server \
  --host 127.0.0.1 \
  --port '"${PORT}"' \
  >"'"${SERVER_LOG}"'" 2>"'"${SERVER_ERR}"'" &
WRAPPER_PID=$!

for i in $(seq 1 900); do
  if ! kill -0 "${WRAPPER_PID}" 2>/dev/null; then
    echo "[ERROR] wrapper exited before health"
    tail -160 "'"${SERVER_LOG}"'" || true
    tail -160 "'"${SERVER_ERR}"'" || true
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:'"${PORT}"'/health" >/tmp/rasst_wrapper_health_'"${SLURM_JOB_ID}"'.json 2>/dev/null; then
    if grep -q "\"status\":\"healthy\"" /tmp/rasst_wrapper_health_'"${SLURM_JOB_ID}"'.json; then
      echo "[INFO] wrapper healthy after ${i}s"
      cat /tmp/rasst_wrapper_health_'"${SLURM_JOB_ID}"'.json
      break
    fi
  fi
  if [ $((i % 30)) -eq 0 ]; then
    echo "[INFO] wrapper still loading after ${i}s"
    cat /tmp/rasst_wrapper_health_'"${SLURM_JOB_ID}"'.json || true
    tail -80 "'"${SERVER_ERR}"'" || true
  fi
  sleep 1
done

if ! grep -q "\"status\":\"healthy\"" /tmp/rasst_wrapper_health_'"${SLURM_JOB_ID}"'.json 2>/dev/null; then
  echo "[ERROR] wrapper did not become healthy"
  tail -160 "'"${SERVER_LOG}"'" || true
  tail -160 "'"${SERVER_ERR}"'" || true
  exit 1
fi

if [ "'"${RUN_SMOKE}"'" = "1" ]; then
  echo "[INFO] running 1-session real E2E smoke"
  python scripts/stress_p0_streaming.py \
    --base-url "http://127.0.0.1:'"${PORT}"'" \
    --agent-type RASST \
    --language-pair "'"${LANGUAGE_PAIR}"'" \
    --sessions 1 \
    --duration-sec "${SMOKE_DURATION_SEC:-30}" \
    --drain-sec "${SMOKE_DRAIN_SEC:-120}" \
    --audio-file "'"${AUDIO_FILE}"'" \
    >"'"${SMOKE_LOG}"'" 2>&1
  cat "'"${SMOKE_LOG}"'"
fi

if [ "'"${RUN_FULL_STRESS}"'" = "1" ]; then
  echo "[INFO] running 32-session 5-minute real E2E stress"
  python scripts/stress_p0_streaming.py \
    --base-url "http://127.0.0.1:'"${PORT}"'" \
    --agent-type RASST \
    --language-pair "'"${LANGUAGE_PAIR}"'" \
    --sessions "${STRESS_SESSIONS:-32}" \
    --duration-sec "${STRESS_DURATION_SEC:-300}" \
    --drain-sec "${STRESS_DRAIN_SEC:-180}" \
    --audio-file "'"${AUDIO_FILE}"'" \
    >"'"${STRESS_LOG}"'" 2>&1
  cat "'"${STRESS_LOG}"'"
fi

curl -fsS "http://127.0.0.1:'"${PORT}"'/health" || true
echo "[INFO] E2E script complete"
'

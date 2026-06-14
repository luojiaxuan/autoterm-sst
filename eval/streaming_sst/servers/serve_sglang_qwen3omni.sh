#!/usr/bin/env bash
# sglang-omni server for *pure* Qwen3-Omni-30B-A3B (no-RAG), thinker TP=2.
# Runs the engine directly (the eval hits its OpenAI endpoint), in the same
# Docker image used by scripts/slurm_rasst_sglang_tp2_e2e_aries.sh. Baseline
# behavior: no SGLANG_OMNI_* M1 knobs are set, so this is current sglang-omni.
#
#   GPUS=2,3 PORT=8100 bash eval/streaming_sst/servers/serve_sglang_qwen3omni.sh
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/mnt/taurus/home/jiaxuanluo/rasst-demo}"
SGLANG_OMNI_SRC="${SGLANG_OMNI_SRC:-/mnt/taurus/home/jiaxuanluo/sglang-omni}"
IMAGE="${IMAGE:-frankleeeee/sglang-omni:dev}"
CONTAINER_NAME="${CONTAINER_NAME:-eval_sglang_qwen3omni}"
# Pure (no-RAG) Qwen3-Omni-30B-A3B-Instruct, resolved HF snapshot on Taurus.
MODEL_PATH="${MODEL_PATH:-/mnt/taurus/data2/jiaxuanluo/.cache/huggingface/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695}"
SERVED_NAME="${SERVED_NAME:-qwen3-omni}"
PORT="${PORT:-8100}"
GPUS="${GPUS:-2,3}"
TP_SIZE="${TP_SIZE:-2}"
# Stage->GPU placement (container-local cuda ids, i.e. indices into GPUS).
# Default: thinker TP on 0,1 and audio/image encoder sharing cuda:1 (baseline).
# To give the encoder its OWN GPU (decontention A/B), pass e.g.
#   GPUS=4,5,6 THINKER_GPUS=0,1 ENCODER_GPU=2
THINKER_GPUS="${THINKER_GPUS:-0,1}"
ENCODER_GPU="${ENCODER_GPU:-1}"
# Container sees cuda:0..N-1 (one per GPU in GPUS).
_NGPU="$(echo "${GPUS}" | tr ',' '\n' | grep -c .)"
CUDA_VIS="$(seq -s, 0 $((_NGPU - 1)))"
THINKER_MAX_SEQ_LEN="${THINKER_MAX_SEQ_LEN:-16384}"
RELAY_BACKEND="${RELAY_BACKEND:-shm}"   # shm (default) | nccl | nixl (relay A/B)
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
# Matches the proven canonical config (scripts/slurm_rasst_sglang_tp2_e2e_aries.sh).
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.75}"
SEG_TMP_DIR="${SEG_TMP_DIR:-/dev/shm/remote_omni_eval}"
# P2c (#760): mixed-chunk folds the running decodes into the streaming chunk-
# prefill step, eliminating the prefill/decode duty-cycle stall (hardened A/B:
# +14% seg/s, -6% StreamLAAL_CA, BLEU parity, no low-N regression, cps-robust).
# Default ON. Set ENABLE_MIXED_CHUNK= (empty) to fall back to the old baseline.
ENABLE_MIXED_CHUNK="${ENABLE_MIXED_CHUNK-1}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-8192}"
MIXED_CHUNK_ARGS=""
[ -n "${ENABLE_MIXED_CHUNK}" ] && MIXED_CHUNK_ARGS="--enable-mixed-chunk"
[ "${CHUNKED_PREFILL_SIZE}" -gt 0 ] 2>/dev/null && \
  MIXED_CHUNK_ARGS="${MIXED_CHUNK_ARGS} --chunked-prefill-size ${CHUNKED_PREFILL_SIZE}"
# M2 de-GIL experiment (#760): set PER_STAGE_PROCESSES=1 to run each non-thinker
# stage in its own OS process (vs one shared GIL-bound "pipeline" process).
# Default off = baseline.
PER_STAGE_PROCESSES="${PER_STAGE_PROCESSES:-}"
SPLIT_ARGS=""
[ -n "${PER_STAGE_PROCESSES}" ] && SPLIT_ARGS="--per-stage-processes"

mkdir -p "${SEG_TMP_DIR}"
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

# Forward any SGLANG_OMNI_* knob set in the launch env into the container
# (e.g. M1 prefill-coalesce: SGLANG_OMNI_PREFILL_COALESCE_MS, *_MIN, *_TARGET,
# *_MAX_TOKENS, SGLANG_OMNI_TP_IDLE_BROADCAST_SKIP, SGLANG_OMNI_LOG_PREFILL_STATS).
# Unset => stock baseline behavior. We pass -e NAME (value taken from this env).
EXTRA_ENV=()
while IFS='=' read -r _k _v; do
  case "${_k}" in
    SGLANG_OMNI_STARTUP_TIMEOUT) : ;;            # already forwarded explicitly
    SGLANG_OMNI_*) EXTRA_ENV+=(-e "${_k}") ;;
  esac
done < <(env)

echo "[sglang] image=${IMAGE} model=${MODEL_PATH} gpus=${GPUS} tp=${TP_SIZE} port=${PORT}"
echo "[sglang] src=${SGLANG_OMNI_SRC} forwarded_knobs=${EXTRA_ENV[*]:-<none>}"

# NOTE: do NOT pass --privileged here. --privileged bypasses the --gpus device
# cgroup isolation, so CUDA_VISIBLE_DEVICES=0,1 would resolve to *physical* host
# GPUs 0,1 instead of the requested ${GPUS}. On a shared node that silently lands
# the engine on whatever is busy (observed: set_device OOM + bogus
# pre_load_avail_mem). Without --privileged, --gpus device=${GPUS} restricts both
# CUDA and NVML to those GPUs (seen inside as cuda:0,1), which is what we want.
exec docker run --rm \
  --name "${CONTAINER_NAME}" \
  --gpus "\"device=${GPUS}\"" \
  --ipc host --network host --shm-size 64g \
  -v /mnt:/mnt -v /home:/home \
  -w "${REPO_ROOT}" \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VIS}" \
  -e HF_HOME=/mnt/taurus/data2/jiaxuanluo/.cache/huggingface \
  -e XDG_CACHE_HOME=/mnt/taurus/data2/jiaxuanluo/.cache \
  -e TORCH_HOME=/mnt/taurus/data2/jiaxuanluo/.cache/torch \
  -e TOKENIZERS_PARALLELISM=false \
  -e SGLANG_OMNI_STARTUP_TIMEOUT="${SGLANG_OMNI_STARTUP_TIMEOUT:-1800}" \
  ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} \
  "${IMAGE}" \
  bash -lc '
set -euo pipefail
cd '"${REPO_ROOT}"'
export PYTHONPATH="'"${SGLANG_OMNI_SRC}"':${PYTHONPATH:-}"
python -c "import sglang_omni.serve, typer, msgpack, av, qwen_vl_utils, librosa, peft, soundfile" >/dev/null 2>&1 \
  || python -m pip install -q msgpack typer av qwen-vl-utils==0.0.11 librosa==0.11.0 numba==0.63.1 peft==0.13.2 soundfile
python -c "import torch;print(\"[sglang] torch\",torch.__version__,\"cuda\",torch.version.cuda,\"gpus\",torch.cuda.device_count())"
exec python scripts/sglang_omni_qwen3_text_tp_server.py \
  --model-path "'"${MODEL_PATH}"'" \
  --model-name "'"${SERVED_NAME}"'" \
  --pipeline-name qwen3-omni \
  --relay-backend '"${RELAY_BACKEND}"' \
  --host 127.0.0.1 --port '"${PORT}"' \
  --ipc-base-path /tmp/eval_sglang \
  --thinker-tp-size '"${TP_SIZE}"' \
  --thinker-gpus '"${THINKER_GPUS}"' \
  --encoder-gpu '"${ENCODER_GPU}"' \
  --thinker-max-seq-len '"${THINKER_MAX_SEQ_LEN}"' \
  --max-running-requests '"${MAX_RUNNING_REQUESTS}"' \
  --max-prefill-tokens '"${MAX_PREFILL_TOKENS}"' \
  --mem-fraction-static '"${MEM_FRACTION_STATIC}"' '"${MIXED_CHUNK_ARGS}"' '"${SPLIT_ARGS}"'
'

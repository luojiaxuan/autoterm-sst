#!/usr/bin/env bash
# LEGACY launcher: standalone InfiniSST scheduler server (serve.api).
#
# Superseded by the thin framework (`start_demo.sh` -> `python -m framework.server`),
# which serves InfiniSST through `framework.agents.infinisst.InfiniSSTAgent`.
# Kept for direct/standalone use and A/B comparison. This is the verbatim
# pre-framework `start_demo.sh` command.
set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/rasst-demo"
PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/infinisst/bin/python}"
FAIRSEQ_ROOT="${FAIRSEQ_ROOT:-/mnt/taurus/data2/jiaxuanluo/fairseq-0.12.2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export PYTHONPATH="${REPO_ROOT}:${FAIRSEQ_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

export RASST_DEMO_MOCK="${RASST_DEMO_MOCK:-0}"

if [ "${RASST_DEMO_MOCK}" != "1" ]; then
  unset RASST_DEMO_FAKE_GPUS
elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -z "${RASST_DEMO_FAKE_GPUS:-}" ]; then
  export RASST_DEMO_FAKE_GPUS="0"
fi

cd "${REPO_ROOT}"

echo "[legacy] Starting InfiniSST scheduler server (serve.api)"
echo "  host: ${HOST}"
echo "  port: ${PORT}"
echo "  python: ${PYTHON_BIN}"
echo "  fairseq: ${FAIRSEQ_ROOT}"
echo "  mock mode: ${RASST_DEMO_MOCK}"
echo "  fake GPUs: ${RASST_DEMO_FAKE_GPUS:-unset}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

exec "${PYTHON_BIN}" -m serve.api \
  --host "${HOST}" \
  --port "${PORT}" \
  --latency-multiplier 2 \
  --min-start-sec 0 \
  --w2v2-path /mnt/aries/data6/xixu/demo/wav2_vec_vox_960h_pl.pt \
  --w2v2-type w2v2 \
  --ctc-finetuned True \
  --length-shrink-cfg "[(1024,2,2)] * 2" \
  --block-size 48 \
  --max-cache-size 576 \
  --model-type w2v2_qwen25 \
  --rope 1 \
  --audio-normalize 0 \
  --max-llm-cache-size 1000 \
  --always-cache-system-prompt \
  --max-len-a 10 \
  --max-len-b 20 \
  --max-new-tokens 10 \
  --beam 4 \
  --no-repeat-ngram-lookback 100 \
  --no-repeat-ngram-size 5 \
  --repetition-penalty 1.2 \
  --suppress-non-language \
  --model-name /mnt/aries/data6/jiaxuanluo/Qwen2.5-7B-Instruct \
  --lora-rank 32

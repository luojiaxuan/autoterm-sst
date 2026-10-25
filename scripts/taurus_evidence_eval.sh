#!/usr/bin/env bash
# Stream the ten-talk alternating ACL/ESO playlist through one session, with the client
# settings of the audited 2026-07-11 runs.
# Usage: taurus_evidence_eval.sh <port> <preset> <run_name> [oracle]
set -euo pipefail

PORT="$1"
PRESET="$2"
NAME="$3"
ORACLE="${4:-}"

ROOT=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
EXTRA=()
if [[ "${PRESET}" != "auto_working" ]]; then
  EXTRA+=(--allow-missing-router-meta)
fi
if [[ "${ORACLE}" == "oracle" ]]; then
  EXTRA+=(--oracle-preset-map nlp=nlp_core_1k,medicine=medicine_core_1k)
fi

cd "${ROOT}/code"
exec "${PYTHON_BIN}" -u eval/streaming_sst/eval_mixed_audio_switch.py \
  --base-url "http://127.0.0.1:${PORT}" \
  --acl-root /mnt/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments \
  --medicine-audio-dir /mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine \
  --acl-items 5 \
  --medicine-items 5 \
  --medicine-ids 404,606,545006,596001,605000 \
  --schedule alternating \
  --seed 20260707 \
  --preset "${PRESET}" \
  --language-pair 'English -> Chinese' \
  --latency-multiplier 2 \
  --base-segment-sec 0.96 \
  --feed-sleep 0 \
  --max-unacked-chunks 1 \
  --backpressure-stall-timeout-sec 60 \
  --idle-timeout-sec 5 \
  --idle-timeouts-after-eof 1 \
  --max-switch-events 20 \
  --no-assert \
  "${EXTRA[@]}" \
  --out-json "${ROOT}/runs/${NAME}.json" \
  --out-md "${ROOT}/runs/${NAME}.md"

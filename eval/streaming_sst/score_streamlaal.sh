#!/usr/bin/env bash
# Score a SimulEval instances.log for BLEU / StreamLAAL / StreamLAAL_CA using
# FBK-fairseq's stream_laal_term.py DIRECTLY (no glossary -> no-RAG, no term
# machinery). Works for either protocol as long as the instances.log wav
# basenames match the audio.yaml wav basenames.
#
#   bash eval/streaming_sst/score_streamlaal.sh \
#     --instances OUT/instances.log \
#     --audio-yaml DATA/audio.yaml --ref DATA/ref.txt --source DATA/source_text.txt
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/mnt/taurus/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python}"
FBK_ROOT="${FBK_ROOT:-/mnt/taurus/home/jiaxuanluo/FBK-fairseq}"
STREAM_LAAL_TOOL="${STREAM_LAAL_TOOL:-${FBK_ROOT}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal_term.py}"
export MWERSEGMENTER_ROOT="${MWERSEGMENTER_ROOT:-/mnt/taurus/home/jiaxuanluo/mwerSegmenter}"
SACREBLEU_TOKENIZER="${SACREBLEU_TOKENIZER:-zh}"
LATENCY_UNIT="${LATENCY_UNIT:-char}"

INSTANCES="" ; AUDIO_YAML="" ; REF="" ; SOURCE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --instances) INSTANCES="$2"; shift 2 ;;
    --audio-yaml) AUDIO_YAML="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --tokenizer) SACREBLEU_TOKENIZER="$2"; shift 2 ;;
    --latency-unit) LATENCY_UNIT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$INSTANCES" && -n "$AUDIO_YAML" && -n "$REF" ]] || { echo "need --instances --audio-yaml --ref" >&2; exit 2; }

echo "[score] instances=${INSTANCES}"
echo "[score] audio_yaml=${AUDIO_YAML} ref=${REF} src=${SOURCE:-<none>}"
echo "[score] tokenizer=${SACREBLEU_TOKENIZER} latency_unit=${LATENCY_UNIT} mwer=${MWERSEGMENTER_ROOT}"

exec "${PYTHON_BIN}" "${STREAM_LAAL_TOOL}" \
  --simuleval-instances "${INSTANCES}" \
  --reference "${REF}" \
  ${SOURCE:+--source-reference "${SOURCE}"} \
  --audio-yaml "${AUDIO_YAML}" \
  --sacrebleu-tokenizer "${SACREBLEU_TOKENIZER}" \
  --latency-unit "${LATENCY_UNIT}"

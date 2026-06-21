#!/usr/bin/env bash
# Build zero-setup working-glossary slices under the Taurus runtime root.
#
# This script builds the CPU-side JSON/manifest artifacts. Build the MaxSim
# indexes with the RASST retriever command printed by build_working_glossary.py,
# then rerun with --require-index if you want the publish step to enforce them.

set -euo pipefail

REPO_ROOT="/mnt/taurus/home/jiaxuanluo/rasst-demo"
TERM_MEMORY_ROOT="${RASST_DEMO_TERM_MEMORY_ROOT:-/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory}"
TARGET_LANG="${TARGET_LANG:-zh}"
LIMIT="${LIMIT:-10000}"
SLICES="${SLICES:-common_10k,nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <translated-glossary.json-or-jsonl> [snapshot-id]" >&2
  exit 2
fi

INPUT="$1"
SNAPSHOT_ID="${2:-working_$(date -u +%Y%m%dT%H%M%SZ)}"

cd "${REPO_ROOT}"
python scripts/term_memory/build_working_glossary.py \
  --input "${INPUT}" \
  --target-lang "${TARGET_LANG}" \
  --slices "${SLICES}" \
  --limit "${LIMIT}" \
  --root "${TERM_MEMORY_ROOT}" \
  --snapshot-id "${SNAPSHOT_ID}"

cat <<EOF

Next steps:
1. Build one MaxSim index per slice, for example:
   python /mnt/taurus/data2/jiaxuanluo/RASST/retriever/build_maxsim_index.py \\
     --model-path /mnt/taurus/home/jiaxuanluo/rasst-demo/checkpoints/retriever/rasst-hn1024.pt \\
     --glossary-path ${TERM_MEMORY_ROOT}/glossaries/common_10k.${TARGET_LANG}.json \\
     --output-path ${TERM_MEMORY_ROOT}/indexes/common_10k/en-${TARGET_LANG}/maxsim.pt \\
     --device cuda:0

2. Serve with:
   export RASST_TERM_MEMORY_MANIFEST=${TERM_MEMORY_ROOT}/manifests/current.json
   export RASST_AUTO_GLOSSARY_ENABLED=1
EOF

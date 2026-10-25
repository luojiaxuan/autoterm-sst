#!/usr/bin/env bash
# note (luojiaxuan): Score one finished ten-talk run with the audited 2026-07-11 pipeline:
# note (luojiaxuan): StreamLAAL bundle -> mwerSegmenter resegmentation + corpus BLEU -> per-domain masked BLEU.
# note (luojiaxuan): Usage: taurus_evidence_score.sh <run_name>   (reads runs/<run_name>.json under the IO root)
set -euo pipefail

NAME="$1"
IO_ROOT=/home/jiaxuanluo/autoterm_taurus_20261024_io
CODE=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/code
JULY=/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711
ACL=/mnt/data2/jiaxuanluo/RASST/data/main_result/inputs/acl_zh
MEDICINE=/mnt/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh
GLOSSARIES=/mnt/data2/jiaxuanluo/RASST/data/glossaries
PYTHON_BIN=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
BUNDLE="${IO_ROOT}/bundles/${NAME}"
SCORES="${IO_ROOT}/scores/${NAME}"
mkdir -p "${SCORES}"

cd "${CODE}"
"${PYTHON_BIN}" eval/streaming_sst/materialize_mixed_streamlaal.py export \
  --run-json "${IO_ROOT}/runs/${NAME}.json" \
  --acl-audio-yaml "${ACL}/audio.yaml" \
  --acl-source "${ACL}/source_text.txt" \
  --acl-reference "${ACL}/ref.txt" \
  --medicine-input-dir "${MEDICINE}" \
  --mask-glossary "${GLOSSARIES}/acl6060_tagged_gt_raw_min_norm2.json" \
  --mask-glossary "${GLOSSARIES}/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json" \
  --out-dir "${BUNDLE}"

"${PYTHON_BIN}" "${JULY}/code/score_merged_realistic_glossary.py" \
  --instances-log "${BUNDLE}/instances.log" \
  --source-file "${BUNDLE}/source.txt" \
  --reference-file "${BUNDLE}/ref.txt" \
  --audio-manifest "${BUNDLE}/audio.json" \
  --glossary "${BUNDLE}/raw_mask_glossary.json" \
  --target-language zh \
  --latency-unit char \
  --sacrebleu-tokenizer zh \
  --mwer-segmenter "${JULY}/bin/mwerSegmenter" \
  --strip-output-tags none \
  --output-tsv "${SCORES}/bleu.tsv" \
  --output-json "${SCORES}/bleu.json" \
  --resegmented-jsonl "${SCORES}/segments.jsonl"

"${PYTHON_BIN}" eval/streaming_sst/score_resegmented_masked_bleu.py \
  --segments-jsonl "${SCORES}/segments.jsonl" \
  --bundle-manifest "${BUNDLE}/manifest.json" \
  --domain-glossary "acl=${GLOSSARIES}/acl6060_tagged_gt_raw_min_norm2.json" \
  --domain-glossary "medicine=${GLOSSARIES}/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json" \
  --output-json "${SCORES}/mt_bleu_domain.json"

cut -f2 "${SCORES}/bleu.tsv" | tail -1 | sed "s/^/${NAME} BLEU=/"
"${PYTHON_BIN}" -c "import json; print('${NAME} MT-BLEU=%.4f' % json.load(open('${SCORES}/mt_bleu_domain.json'))['bleu'])"

#!/usr/bin/env bash
# Score the RealSI ten-domain streams on Taurus: time-aligned TERM_ACC per stream, pooled BLEU with paired
# talk-level tests of the language-model router against every other configuration, and the live routing
# coverage of both routers. Run JSONs are read from realsi/<name>_s<order>.json under the IO root.
set -euo pipefail
IO=/home/jiaxuanluo/autoterm_taurus_20261024_io
CODE=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/code
PY=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
JULY=/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711
REALSI=/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/realsi_65f3eff/en2zh/json
NAMES=(lm_hmm cosine known merged10k noterm)
ORDERS=(1 2 3)
SLICES=nlp=nlp_core_1k,medicine=medicine_core_1k,education=education_core_1k,finance=finance_core_1k,legal=legal_core_1k,environment=environment_core_1k,entertainment=entertainment_core_1k,science=science_core_1k,sports=sports_core_1k,art=art_core_1k
cd "$CODE"
RUNS=(); SYSTEMS=(); ROUTED=()
for n in "${NAMES[@]}"; do
  files=()
  for o in "${ORDERS[@]}"; do
    RUNS+=(--run "${n}_s$o=$IO/realsi/${n}_s$o.json")
    files+=("$IO/realsi/${n}_s$o.json")
    [[ $n == lm_hmm || $n == cosine ]] && ROUTED+=(--run "${n}_s$o=$IO/realsi/${n}_s$o.json")
  done
  SYSTEMS+=(--system "$n=$(IFS=,; echo "${files[*]}")")
done
$PY eval/streaming_sst/score_realsi_terms.py --repo . --terms docs/realsi_20261120/realsi_terms.json "${RUNS[@]}" \
  --out "$IO/analysis/realsi_terms.json"
$PY eval/streaming_sst/evidence/routing_coverage.py "${ROUTED[@]}" --slice-map "$SLICES" \
  --group lm_hmm=lm_hmm_s1,lm_hmm_s2,lm_hmm_s3 --group cosine=cosine_s1,cosine_s2,cosine_s3 \
  --out "$IO/analysis/realsi_routing_coverage.json"
$PY eval/streaming_sst/evidence/realsi_eval.py "${SYSTEMS[@]}" --term-hits "$IO/analysis/realsi_terms.json" \
  --realsi-json "$REALSI" --bleu-scorer "$JULY/code/score_merged_realistic_glossary.py" \
  --mwer-segmenter "$JULY/bin/mwerSegmenter" --pair lm_hmm,known --pair lm_hmm,cosine --pair lm_hmm,merged10k --pair lm_hmm,noterm \
  --pair cosine,known --out "$IO/analysis/realsi_eval.json"
echo scored

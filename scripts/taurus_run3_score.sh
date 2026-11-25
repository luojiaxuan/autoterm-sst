#!/usr/bin/env bash
# Score the third ten-talk run (two H200 GPUs, 2026-11-20) on Taurus with the audited pipeline: BLEU and
# MT-BLEU per run, per-occurrence TERM_ACC for all five runs, prompt precision, and paired talk-level tests
# of the language-model router against every other configuration. Run JSONs are read from
# runs/r3_<name>.json under the IO root.
set -euo pipefail
IO=/home/jiaxuanluo/autoterm_taurus_20261024_io
CODE=/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/code
PY=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
GOLD=/mnt/data2/jiaxuanluo
GLOSS=$GOLD/RASST/data/glossaries
NAMES=(lm_hmm cosine known merged2k merged10k)
cd "$CODE"
for n in "${NAMES[@]}"; do
  [[ -s "$IO/scores/r3_$n/mt_bleu_domain.json" ]] || bash scripts/taurus_evidence_score.sh "r3_$n" | tail -2
done
RUNS=(); DATA=()
for n in "${NAMES[@]}"; do
  RUNS+=(--run "$n=$IO/runs/r3_$n.json")
  DATA+=(--run-data "$n=$IO/scores/r3_$n/segments.jsonl,$IO/bundles/r3_$n/manifest.json")
done
$PY eval/streaming_sst/evidence/per_occurrence_terms.py --repo . --gold-root "$GOLD" "${RUNS[@]}" \
  --out "$IO/analysis/r3_term_occurrences.jsonl"
$PY eval/streaming_sst/score_prompt_precision.py "${RUNS[@]}" --mfa-root eval/streaming_sst/mfa_alignments \
  --acl-root "$GOLD/rasst_eval/acl6060_zh_segments" --acl-technical-gold eval/streaming_sst/acl_gold_technical.json \
  --acl-raw-glossary "$GLOSS/acl6060_tagged_gt_raw_min_norm2.json" \
  --medicine-oracle-dir "$GOLD/RASST/data/main_result/inputs/medicine_zh" \
  --out-json "$IO/analysis/r3_prompt_precision.json" > /dev/null
PAIRS=(--pair lm_hmm,known --pair lm_hmm,cosine --pair lm_hmm,merged2k --pair lm_hmm,merged10k --pair cosine,known)
$PY eval/streaming_sst/evidence/paired_significance.py --repo . --gloss-dir "$GLOSS" "${DATA[@]}" "${PAIRS[@]}" \
  --occurrences "$IO/analysis/r3_term_occurrences.jsonl" --n-boot 4000 --seed 20261024 \
  --out "$IO/analysis/r3_paired_significance.json" > /dev/null
$PY eval/streaming_sst/evidence/term_free_quality.py --repo . --gloss-dir "$GLOSS" "${DATA[@]}" "${PAIRS[@]}" \
  --n-boot 4000 --out "$IO/analysis/r3_term_free_quality.json" > /dev/null
echo scored

#!/usr/bin/env bash
# AutoTerm-SST of run 3 paired with the Merged-100k and Merged-1M streams of run 1: same audio order and chunk
# boundaries, so the talk-level tests pair them directly.
set -euo pipefail
cd /mnt/data2/jiaxuanluo/autoterm_taurus_20261024/code
PY=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
GOLD=/mnt/data2/jiaxuanluo; GLOSS=$GOLD/RASST/data/glossaries
RUNS=(--run lm_hmm=/home/jiaxuanluo/autoterm_taurus_20261024_io/runs/r3_lm_hmm.json --run merged100k=/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/runs/merged100k.json --run merged1m=/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/runs/merged1m.json)
DATA=(--run-data lm_hmm=/home/jiaxuanluo/autoterm_taurus_20261024_io/scores/r3_lm_hmm/segments.jsonl,/home/jiaxuanluo/autoterm_taurus_20261024_io/bundles/r3_lm_hmm/manifest.json
      --run-data merged100k=/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/scores/merged100k/segments.jsonl,/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/bundles/merged100k/manifest.json
      --run-data merged1m=/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/scores/merged1m/segments.jsonl,/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/bundles/merged1m/manifest.json)
PAIRS=(--pair lm_hmm,merged100k --pair lm_hmm,merged1m)
$PY eval/streaming_sst/evidence/per_occurrence_terms.py --repo . --gold-root $GOLD "${RUNS[@]}" --out /home/jiaxuanluo/autoterm_taurus_20261024_io/analysis/r3x1_term_occurrences.jsonl
$PY eval/streaming_sst/evidence/paired_significance.py --repo . --gloss-dir $GLOSS "${DATA[@]}" "${PAIRS[@]}" --occurrences /home/jiaxuanluo/autoterm_taurus_20261024_io/analysis/r3x1_term_occurrences.jsonl --n-boot 4000 --seed 20261024 --out /home/jiaxuanluo/autoterm_taurus_20261024_io/analysis/r3x1_paired_significance.json > /dev/null
$PY eval/streaming_sst/evidence/term_free_quality.py --repo . --gloss-dir $GLOSS "${DATA[@]}" "${PAIRS[@]}" --n-boot 4000 --out /home/jiaxuanluo/autoterm_taurus_20261024_io/analysis/r3x1_term_free_quality.json > /dev/null
echo scored

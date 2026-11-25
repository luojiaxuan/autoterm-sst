#!/usr/bin/env bash
# Score the equal-catalog runs of drive_aries.sh with the protocol of scripts/taurus_run3_score.sh. Per arm:
# StreamLAAL bundle, mWER-resegmented BLEU and MT-BLEU (the steps of scripts/taurus_evidence_score.sh). Per
# scale: per-occurrence TERM_ACC, prompt precision, paired talk-level tests (exact permutation over the 2^10
# talk swaps, domain-stratified talk bootstrap, Holm over the scale's nine contrasts) and BLEU on sentences
# whose reference holds no glossary target, for routed vs merged, routed vs known and merged vs known; then
# summary.json with every arm's absolute values, overall and per domain.
# CODE is the scoring tree of the paper's runs: its scorers are identical to this revision's, and it also
# holds the ESO medicine TextGrids under eval/streaming_sst/mfa_alignments that Git does not carry.
#   score.sh <run root> <wrapper revision> [scale ...]     (default 1m 100k; writes <run root>/<scale>/scores/)
set -euo pipefail
ROOT=${1:?run root}
REV=${2:?wrapper revision}
shift 2
SCALES=("${@:-1m}")
[[ $# -gt 0 ]] || SCALES=(1m 100k)
CODE=${CODE:-/mnt/taurus/data2/jiaxuanluo/autoterm_taurus_20261024/code}
PY=/home/jiaxuanluo/miniconda3/envs/spaCyEnv/bin/python
JULY=/mnt/taurus/data1/jiaxuanluo/autoterm_streamlaal_20260711
GOLD=/mnt/taurus/data2/jiaxuanluo
GLOSS=$GOLD/RASST/data/glossaries
INPUTS=$GOLD/RASST/data/main_result/inputs
ACL_GLOSS=$GLOSS/acl6060_tagged_gt_raw_min_norm2.json
MED_GLOSS=$GLOSS/hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json
ARMS=(routed merged known)
PAIRS=(--pair routed,merged --pair routed,known --pair merged,known)
cd "$CODE"

arm() {  # arm <scale> <arm>
  local s=$ROOT/$1/scores/$2
  [[ -s $s/mt_bleu_domain.json ]] && return 0
  mkdir -p "$s"
  "$PY" eval/streaming_sst/materialize_mixed_streamlaal.py export --run-json "$ROOT/$1/out/$2.json" \
    --acl-audio-yaml "$INPUTS/acl_zh/audio.yaml" --acl-source "$INPUTS/acl_zh/source_text.txt" \
    --acl-reference "$INPUTS/acl_zh/ref.txt" --medicine-input-dir "$INPUTS/medicine_zh" \
    --mask-glossary "$ACL_GLOSS" --mask-glossary "$MED_GLOSS" --out-dir "$s/bundle"
  "$PY" "$JULY/code/score_merged_realistic_glossary.py" --instances-log "$s/bundle/instances.log" \
    --source-file "$s/bundle/source.txt" --reference-file "$s/bundle/ref.txt" --audio-manifest "$s/bundle/audio.json" \
    --glossary "$s/bundle/raw_mask_glossary.json" --target-language zh --latency-unit char --sacrebleu-tokenizer zh \
    --mwer-segmenter "$JULY/bin/mwerSegmenter" --strip-output-tags none --output-tsv "$s/bleu.tsv" \
    --output-json "$s/bleu.json" --resegmented-jsonl "$s/segments.jsonl"
  "$PY" eval/streaming_sst/score_resegmented_masked_bleu.py --segments-jsonl "$s/segments.jsonl" \
    --bundle-manifest "$s/bundle/manifest.json" --domain-glossary "acl=$ACL_GLOSS" \
    --domain-glossary "medicine=$MED_GLOSS" --output-json "$s/mt_bleu_domain.json"
}

scale() {  # scale <scale>
  local S=$ROOT/$1/scores pids=() RUNS=() DATA=() a p
  mkdir -p "$S"
  for a in "${ARMS[@]}"; do
    arm "$1" "$a" > "$S/$a.log" 2>&1 & pids+=($!)
    RUNS+=(--run "$a=$ROOT/$1/out/$a.json")
    DATA+=(--run-data "$a=$S/$a/segments.jsonl,$S/$a/bundle/manifest.json")
  done
  [[ -s $S/term_occurrences.jsonl ]] || { "$PY" eval/streaming_sst/evidence/per_occurrence_terms.py --repo . \
    --gold-root "$GOLD" "${RUNS[@]}" --out "$S/term_occurrences.jsonl" > "$S/term_occurrences.log" 2>&1 & pids+=($!); }
  [[ -s $S/prompt_precision.json ]] || { "$PY" eval/streaming_sst/score_prompt_precision.py "${RUNS[@]}" \
    --mfa-root eval/streaming_sst/mfa_alignments --acl-root "$GOLD/rasst_eval/acl6060_zh_segments" \
    --acl-technical-gold eval/streaming_sst/acl_gold_technical.json --acl-raw-glossary "$ACL_GLOSS" \
    --medicine-oracle-dir "$INPUTS/medicine_zh" --out-json "$S/prompt_precision.json" > "$S/prompt_precision.log" 2>&1 &
    pids+=($!); }
  for p in "${pids[@]}"; do wait "$p"; done
  pids=()
  [[ -s $S/paired_significance.json ]] || { "$PY" eval/streaming_sst/evidence/paired_significance.py --repo . \
    --gloss-dir "$GLOSS" "${DATA[@]}" "${PAIRS[@]}" --occurrences "$S/term_occurrences.jsonl" --n-boot 4000 \
    --seed 20261024 --out "$S/paired_significance.json" > "$S/paired_significance.log" 2>&1 & pids+=($!); }
  [[ -s $S/term_free_quality.json ]] || { "$PY" eval/streaming_sst/evidence/term_free_quality.py --repo . \
    --gloss-dir "$GLOSS" "${DATA[@]}" "${PAIRS[@]}" --n-boot 4000 --out "$S/term_free_quality.json" \
    > "$S/term_free_quality.log" 2>&1 & pids+=($!); }
  for p in "${pids[@]}"; do wait "$p"; done
  "$PY" - "$S" "$ACL_GLOSS" "$MED_GLOSS" "$ROOT" "$1" "$REV" "$CODE" > "$S/summary.json" <<'PY'
import json, sys, time
from pathlib import Path
import numpy as np
import sacrebleu
sys.path[:0] = [".", "eval/streaming_sst/evidence"]
from paired_significance import Corpus, holm
from eval.streaming_sst.score_terms import load_target_terms_for_masking

S, acl_gloss, med_gloss, root, scale, rev, code = sys.argv[1:8]
S = Path(S)
arms = ("routed", "merged", "known")
terms = {"acl": load_target_terms_for_masking(acl_gloss, "zh"), "medicine": load_target_terms_for_masking(med_gloss, "zh")}
occ = [json.loads(line) for line in open(S / "term_occurrences.jsonl", encoding="utf-8")]
odom = np.array([{"nlp": "acl"}.get(o["domain"], o["domain"]) for o in occ])
assert set(odom) == {"acl", "medicine"}, set(odom)
pp = json.load(open(S / "prompt_precision.json"))["runs"]
sig = json.load(open(S / "paired_significance.json"))["contrasts"]
tf = json.load(open(S / "term_free_quality.json"))
out = {
    "meta": {"run_root": root, "scale": scale, "run_revision": json.load(open(f"{root}/{scale}/manifest.json"))["revision"],
             "wrapper_revision": rev, "scorer_tree": code, "sacrebleu": sacrebleu.__version__,
             "scored_at": time.strftime("%FT%TZ", time.gmtime()),
             "term_acc": "time-aligned per-occurrence hits over the raw_plus_medicine gold, in percent",
             "term_free_bleu": "corpus BLEU over sentences whose reference contains no target term of its domain glossary"},
    "arms": {}, "contrasts": {},
}
for arm in arms:
    manifest = json.load(open(S / arm / "bundle" / "manifest.json"))
    corpus_of_wav = {b["wav"]: b["corpus"] for b in manifest["blocks"]}
    rows = [json.loads(line) for line in open(S / arm / "segments.jsonl", encoding="utf-8") if line.strip()]
    corp = Corpus([r["prediction"] for r in rows], [r["reference"] for r in rows])
    dom = np.array([corpus_of_wav[r["wav"]] for r in rows])
    free = np.array([not any(t and t in r["reference"] for t in terms[d]) for r, d in zip(rows, dom)])
    hit = np.array([o[f"hit_{arm}"] for o in occ], dtype=float)
    entry = {
        "term_acc": 100 * hit.mean(), "occurrences": len(occ), "bleu": corp.full,
        "bleu_scorer_json": json.load(open(S / arm / "bleu.json"))["BLEU"],
        "term_free_bleu": corp.score(np.flatnonzero(free)), "sentences": len(rows), "term_free_sentences": int(free.sum()),
        "mt_bleu": json.load(open(S / arm / "mt_bleu_domain.json"))["bleu"],
        "prompt_precision": 100 * pp[arm]["raw_plus_medicine"]["prompt_precision"],
        "references_per_chunk": pp[arm]["retrieved_references_per_chunk"], "by_domain": {},
    }
    for d in ("acl", "medicine"):
        entry["by_domain"][d] = {
            "term_acc": 100 * hit[odom == d].mean(), "occurrences": int((odom == d).sum()),
            "bleu": corp.score(np.flatnonzero(dom == d)), "sentences": int((dom == d).sum()),
            "term_free_bleu": corp.score(np.flatnonzero((dom == d) & free)), "term_free_sentences": int(((dom == d) & free).sum()),
        }
    out["arms"][arm] = entry
for a, b in (("routed", "merged"), ("routed", "known"), ("merged", "known")):
    c = {m: sig[f"{a}_vs_{b}:{m}"] for m in ("term_acc", "bleu", "mt_bleu")}
    c["term_free_bleu"] = tf[f"{a}_vs_{b}"]
    c["holm_over_term_acc_and_bleu"] = holm({m: c[m]["talk_permutation_p"] for m in ("term_acc", "bleu")})
    out["contrasts"][f"{a}_vs_{b}"] = c
print(json.dumps(out, indent=1, ensure_ascii=False))
PY
}

pids=()
for s in "${SCALES[@]}"; do scale "$s" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
echo scored

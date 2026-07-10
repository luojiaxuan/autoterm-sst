# Multi-domain routing exploration

Branch `explore/multidomain-routing`. Goal: falsify (cheaply) whether the
training-free router stays correct when distractor domains are registered,
before investing in a 4×100k / merged-400k end-to-end run.

## Diagnosis of the current router (confirmed)

- The production keyword scorer `topic_keyword_scores` reads
  `DOMAIN_TOPIC_KEYWORDS`, which registers **4 domains** (nlp, medicine,
  finance, legal). The 10-domain `DOMAIN_KEYWORDS` list is *not* used by the
  live topic channel. So the router is genuinely 4-domain-capable today.
- `topic_router.py` computes an EMA-smoothed `ema_final_score` per window
  (line ~873) and logs it, but the ranking sort uses `confidence = raw_final`
  (lines ~880, ~903). **The EMA smoothing is dead code w.r.t. the decision.**
  This is a candidate fix if multi-domain switching proves jittery.
- Domain probe cost is linear in the candidate count (one MaxSim query per
  candidate index per refresh; `omni.py` `_domain_probe_slices`).

## Stage 1 — keyword-channel falsification (CPU, done)

`falsify_keywords.py` scores the REAL generated-target output of the zh
ten-talk run against the 4 registered domains in sliding windows.

| true domain | signal windows | top-1 correct | distractor top-1 |
|---|---:|---:|---:|
| nlp (ACL)   | 104 | 0.981 | 2/104 (both to medicine) |
| medicine    | 701 | 0.999 | 1/701 |

finance/legal **never win top-1** on ACL windows; on medicine windows legal
fires a positive score on 15/701 and finance on 5/701 but neither wins. The
dominant 0.60 signal is robust to the finance/legal distractors. **PASS.**

## Stage 2 — probe-channel + E2E (in progress)

Carved representative finance (2,071) and legal (5,397) slices from the 1M wiki
pool by domain keywords (`carve_domains.py`); built their MaxSim indexes.
Re-ran a 3-talk ACL/medicine stream against a 4-candidate host (8014) to test
the full router (keyword + audio probe) for mis-switches to finance/legal.
Results appended below when the run completes.

## Scope note

The 1M wiki pool only yields 2–5k domain-matching terms, so the user's
100k-per-domain / merged-400k end-to-end comparison needs a proper Wikidata
P31/subclass extraction (stage 3), not a keyword carve. Do that only if
Stage 2 routing passes.

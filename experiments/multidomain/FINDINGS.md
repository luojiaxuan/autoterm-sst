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

## Stage 2 result — 4-way E2E routing (PASS, and informative)

3-talk ACL/medicine stream against a 4-candidate host (nlp/medicine/finance/legal):

| metric | value |
|---|---|
| wrong switches | **0** |
| finance/legal active-domain events | **0 / 1997** |
| steady-state active-domain accuracy | **1.0** |
| transitions | 2/2 correct (nlp→med 44.3s, med→nlp 27.7s) |
| domain-probe p95 latency (4 candidates) | 3.2 ms |
| retrieval p50/p95 | 79.9 / 106.6 ms |

**The informative part:** the raw audio probe channel *did* rank a distractor
top in 318/1997 windows (legal 247, finance 71 — ~16%), yet the router never
switched to them. This is a direct demonstration of the intended design: the
noisy audio probe (0.25) is out-voted by the keyword-led topic signal (0.60)
and suppressed by the consistency/margin/confidence guards. "Probe as guard,
not leader" holds empirically at 4 domains.

## Go / no-go

Routing **passes** on both channels at 4 domains, and the probe-noise-but-no-
misswitch result is a genuine paper point ("a simple training-free router
selects correctly among registered terminology resources, robust to a noisy
audio probe"). This justifies Stage 3.

## Stage 3 (justified, not yet run)

Proper Wikidata P31/subclass extraction for tech/medicine/law/finance at 100k
each (the 1M wiki pool only yields 2–5k domain terms by keyword carve), then
the end-to-end 3-talk comparison the user asked for:
oracle-domain vs AutoTerm-routed vs merged-400k-flat, reporting MFA term_acc
and BLEU. Expected story: routed slices keep retrieval precision that the flat
400k merge loses (cf. the scale sweep in the paper), so AutoTerm ≈ oracle while
merged-400k degrades.

The `ema_final` dead-code path is a latent risk if more domains are added; not
triggered at 4 domains but worth wiring into the sort before scaling to 10.

## Sizing correction (2026-07-10)

The Stage-2 4-way routing test used these ACTUAL slice sizes (the
`*_wiki_100k` filenames were aspirational, not literal):

| domain | terms | source |
|---|---:|---|
| nlp | 9,933 | curated gold union |
| medicine | 9,996 | curated gold union |
| finance | 2,071 | keyword carve from 1M general pool |
| legal | 5,397 | keyword carve from 1M general pool |

Real domain vocabularies are ~10k, not 100k (beyond ~10-30k Wikidata yields
long-tail noise — exactly the precision collapse in the paper's scale sweep).
So Stage 3 targets a MATCHED ~12k per domain (aligned with the paper's 10k
union convention); "merged" becomes a flat 4x12k index vs the routed slices.
The routing conclusion (0 mis-switches) is independent of slice size — it is
about which domain, not how many terms.

finance/legal are being rebuilt from proper Wikidata P31/category terms
(collect_wikimedia_domain_glossary.py) at 12k, replacing the general-pool carve.

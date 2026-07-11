# Multi-domain routing, and a merged-glossary confound worth recording

Exploration branch, folded to main for the record. Goal: test whether the
training-free router stays correct as more terminology domains are registered,
and whether a single flat "merged" glossary can replace routed domain slices.

Two things worth keeping: (1) the router's 4-way falsification passes cleanly,
and (2) a cautionary result — an early "merged hurts German" reading turned out
to be a **glossary-completeness confound**, not a precision effect. Once the
merged glossary is complete in the target language, a flat 42k inventory matches
or beats routing on end-to-end quality at this scale; the routing advantage
lives in retrieval precision and at larger sizes, not here. Recorded so the
mistake isn't repeated.

All runs are 4-talk alternating ACL/medicine streams (2 ACL + 2 medicine:
talks 268/367, medicine 545006/606) at latency multiplier 2, one live session
per condition, on aries A6000s. Term accuracy is reported two ways: **MFA**
(time-aligned occurrence scoring, `eval/streaming_sst/score_time_aligned_terms.py`)
where it is reliable, and **count-clipped**
(`eval/streaming_sst/score_mixed_audio_terms.py`) which also emits BLEU and
masked-BLEU. MFA time-alignment is unreliable for de/ja (the target output
timing does not line up with the source-audio gold windows), so the
cross-lingual comparison uses the count-clipped scorer for all three languages.

## 1. Router diagnosis (confirmed)

- The live keyword scorer (`topic_keyword_scores` → `DOMAIN_TOPIC_KEYWORDS`)
  registers **4 domains**: nlp, medicine, finance, legal. The 10-domain
  `DOMAIN_KEYWORDS` list is not used by the live topic channel.
- `topic_router.py` computes an EMA-smoothed `ema_final_score` per window and
  logs it, but the ranking sorts by `confidence = raw_final` — **the EMA
  smoothing never affects the decision.** Latent risk if domains scale past 4.
- Domain-probe cost is linear in candidate count (one MaxSim query per
  candidate index per refresh).

## 2. Routing falsification (PASS on both channels)

**Keyword channel** (`falsify_keywords.py`, CPU): scoring the real generated
output of the zh ten-talk run against the 4 domains in sliding windows —
ACL windows 98.1% top-1 correct, medicine 99.9%; finance/legal never win top-1.

**4-way E2E** (`results/routing_4way_*.json`): a 3-talk stream against a host
with all 4 domains as live candidates — **0 wrong switches, finance/legal
active on 0/1997 windows, steady-state active-domain accuracy 1.0**, domain
probe p95 3.2 ms. The informative part: the raw audio probe ranked a distractor
#1 on 318/1997 windows (~16%), yet the keyword-led fusion + guards suppressed
every one. "Probe as guard, not leader" holds empirically at 4 domains.

## 3. Multilingual: merged glossary hides its cost in Chinese, exposes it in German

Same 4-talk stream, all conditions at matched 2× rate. `oracle` = each block
scored with its own-domain fixed slice (upper bound); `AutoTerm` = router picks
among the 4 candidates; `merged-42k` = one flat index of 41,667 deduplicated
terms (nlp+medicine+finance+legal), no routing. term_acc / BLEU / masked-BLEU
via the count-clipped scorer:

| Chinese (strong model) | term_acc | BLEU | masked-BLEU |
|---|---:|---:|---:|
| AutoTerm | 0.881 | 57.56 | 55.10 |
| merged-42k | 0.896 (+1.5) | 57.34 (−0.22) | 54.86 (−0.24) |

| German (weaker model, human-verified refs) | term_acc | BLEU | masked-BLEU |
|---|---:|---:|---:|
| AutoTerm | 0.733 | 34.01 | 33.58 |
| merged-42k (finance/legal not de-translated) | 0.793 | 33.38 (−0.63) | 32.79 (−0.79) |
| **merged-42k (de-complete)** | **0.837** | **34.44** (+0.43) | **33.98** (+0.40) |

**Reading (with a self-correction).** A first pass suggested merged hurt German
fluency (masked-BLEU −0.79). That was a **confound, not an effect**: the
first-pass finance/legal glossary had no German, so the merged index injected
English/blank strings into German prompts. Once finance/legal carry German
labels (the de-complete row), the penalty disappears — merged is actually
slightly *better* than AutoTerm on all three German metrics.

So at this 42k scale, a **complete** flat glossary matches or beats routing on
end-to-end term_acc/BLEU/masked-BLEU in both Chinese and German. This 4-talk
end-to-end comparison does **not** expose a merged-vs-routed penalty; the
routing argument (compact slices preserve retrieval precision) lives in
retrieval precision (Prec@10) and at much larger merged sizes — see the paper's
scale sweep where precision collapses 0.48→0.14 as the inventory grows to 100k,
while term accuracy stays flat. The honest reading here is that 42k is not yet
large enough for the flat merge to cost end-to-end quality.

Japanese confirms the same picture, but only after two artifacts were removed.
A parallel 4-condition run first suggested merged *craters* Japanese (BLEU
34.0 vs 22.3). That was **contention, not an effect**: four eval streams on one
host degraded every run's generation quality — AutoTerm itself dropped from a
clean 41.8 BLEU to 34.0, and merged dropped more. Re-run **solo** on a fresh
host:

| Japanese (solo, clean host) | term_acc | BLEU | masked-BLEU |
|---|---:|---:|---:|
| AutoTerm | 0.863 | 41.84 | 40.02 |
| merged-42k | 0.846 | 42.12 (+0.28) | 40.44 (+0.42) |

merged again matches AutoTerm (marginally better on BLEU/masked-BLEU). So all
three languages agree: at 42k a complete flat glossary carries no end-to-end
penalty.

**Method lesson worth keeping:** running several eval streams concurrently on
one host does not just risk 1011 keepalive failures — it silently lowers output
*quality* on the runs that survive, and unevenly. Comparison conditions must be
run one-at-a-time (or on separate hosts) at a matched feed rate, or a
contention artifact masquerades as a real effect. The earlier zh/de parallel
batches happened to degrade evenly; the ja batch did not.

## 4. Slice size: 10k is oversized; ~1k is enough (Chinese)

`oracle` with 1k slices (all gold kept, fillers capped at 1000) vs 10k, MFA
term accuracy:

| zh oracle slice | ACL term_acc | medicine term_acc |
|---|---:|---:|
| 10k | 0.871 | 0.826 |
| **1k** | **0.886** | 0.826 |

Shrinking the NLP slice 10k→1k *raised* ACL term accuracy (+1.5pp) — the extra
~9k wiki fillers dilute retrieval — while BLEU stayed flat (55.8/55.1). The 1k
combined oracle (0.863) even tops merged-42k (0.846). Direct support for the
paper's "compact active slice" argument.

## 5. Data & artifacts

- `glossaries/finance_wiki_12k.json`, `legal_wiki_12k.json` — 12k terms each,
  collected from Wikidata by P31/subclass + Wikipedia category expansion
  (`scripts/collect_wikimedia_domain_glossary.py`). Every row keeps its
  `wikidata_qid`, `domain_root_qid`, and `rdf_path` for provenance. German and
  Japanese labels backfilled from Wikidata `wbgetentities`
  (`scripts/fetch_de_wbget.py`, `fetch_ja_wbget.py`): de ~87%/74%, ja ~90%/82%
  (finance/legal). zh from the original collection.
- `results/` — scored summaries: `mfa_*` (time-aligned term accuracy + traces),
  `bleu_*` (term_acc + BLEU + masked-BLEU), `routing_4way_*` (falsification).
- `scripts/carve_domains.py` — quick keyword carve from the 1M wiki pool (used
  before the proper Wikidata collection).
- **MaxSim indexes are not committed** (`*.pt` is gitignored; the runtime keeps
  indexes on the demo host per the manifest convention). See `INDEXES.md` for
  their cluster paths and the one-line rebuild command from the glossaries.

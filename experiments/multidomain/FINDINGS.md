# Multi-domain routing & the multilingual precision–fluency trade-off

Exploration branch, folded to main for the record. Goal: test whether the
training-free router stays correct as more terminology domains are registered,
and whether a single flat "merged" glossary can replace routed domain slices.
The most transferable result is **multilingual**: a flat merged glossary looks
harmless in Chinese but visibly degrades German.

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
| merged-42k | **0.896** (+1.5) | 57.34 (−0.22) | 54.86 (**−0.24**) |

| German (weaker model, human-verified refs) | term_acc | BLEU | masked-BLEU |
|---|---:|---:|---:|
| AutoTerm | 0.733 | 34.01 | 33.58 |
| merged-42k* | **0.793** (+6.0) | 33.38 (−0.63) | 32.79 (**−0.79**) |

`*` German merged used the first-pass glossary (finance/legal not yet
de-translated); a clean-merged rerun (de coverage 84%) and the Japanese
condition were queued and will be appended when they land.

**Reading:** merged always *raises* term accuracy — with every term available
it retrieves more — but pays in fluency. In Chinese the cost is negligible
(masked-BLEU −0.24); the strong Chinese model shrugs off the 42k of
cross-domain distractors. In German the same merge costs ~3× more
(masked-BLEU −0.79, BLEU −0.63): the weaker model, judged against human
references, is far less robust to a diluted, noisy term map. This is exactly
why routing matters and why a Chinese-only evaluation would miss it — routed
slices keep the compact, precise inventory that a naive merge throws away.

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

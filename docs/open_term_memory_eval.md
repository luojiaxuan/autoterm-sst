# Open terminology memory — scale & latency evaluation

Implementation note (2026-06-19): the scale results below motivate the
zero-setup adaptive working glossary now documented in
[`adaptive_working_glossary.md`](adaptive_working_glossary.md). Large
100k/500k/1M memories remain offline memory and stress evidence; serving
defaults to `auto_working`, which starts from a domain-specific slice such as
`nlp_core_10k`, routes among domain slices, and injects only the fixed
`RASST_PROMPT_TOP_K` references into the prompt.

Streaming SST with the RASST Qwen3-Omni agent (in-process vLLM tp=2 + MaxSim RAG)
on aries (A6000). Audio: ACL 60-60 talk (`acl6060_zh_smoke`, en→zh), streamed
0.5 s/chunk over the framework JSON WebSocket. Latencies from event `meta`
(`retrieve_s`, `elapsed_s`); "warm" excludes the first chunk (which pays the
one-time index load). Tool: `eval/streaming_sst/sweep_term_memory.py`.

## Scale sweep (en→zh)

| preset | terms | warm retrieve p50 / p95 (ms) | gen p50 (ms) | refs / chunk | cold index load |
|---|---:|---:|---:|---:|---:|
| none | 0 | — | 480 | 0 | — |
| acl_tagged_raw (curated) | 238 | 61 / 105 | 643 | 2.33 | ~0.5 s |
| open_wiki_10k (p31 obscure) | 10 000 | 61 / 64 | 450 | 0.33 | ~0.3 s |
| open_wiki_academic (relevant) | 18 994 | 61 / 79 | 524 | 1.90 | ~1.0 s |
| open_wiki_100k (general) | 100 000 | 64 / 83 | 638 | 4.89 | ~6.8 s |
| open_wiki_1m (general) | 1 000 000 | 78 / 95 | 610 | 9.62 | > 30 s (4 GB) |

## Findings

1. **Exact MaxSim retrieval is encoder-bound, not term-count-bound.** Warm
   per-chunk retrieval stays ~60–95 ms p50 from 238 → 1,000,000 terms. The
   similarity matmul over the term matrix is negligible next to the streaming
   audio-encoder pass. **→ Two-stage retrieval (ANN coarse + rerank) is NOT
   required for latency, even at 1M.** It remains future work only if term count
   grows far beyond 1M or GPU memory for the term matrix becomes the limit
   (1M × 1024 fp = ~4 GB on the RAG GPU).

2. **The real scale cost is the one-time cold index load**, which grows with size
   (100k ≈ 6.8 s, 1M > 30 s to `torch.load` + move the 4 GB tensor to GPU). In
   the UI this is paid at preset-selection time (`/glossary/build` pre-activates
   the index), so streaming is warm. For eval, warm the index before timing.
   *Improvement:* pre-warm the active/default open-memory index at startup.

3. **Curated precision vs open recall.** The curated ACL glossary (238) yields
   2.33 refs/chunk and the model used 19 marked terms on the in-domain talk; the
   relevant academic 19k used 2; the obscure p31 10k used 1. Larger general
   indexes raise refs/chunk (100k: 4.89, 1M: 9.62) but with lower precision /
   more noise. **→ A domain-adaptive active slice (~10–30 k high-quality terms)
   is the right front-end setting**; 100k/1M is the background scalability story.

4. **High-quality domain-relevant Wikidata terms (with zh) are limited.** Keyword
   filtering the 12.4M translated glossary yields ~19k clean academic terms; open
   Wikidata is dominated by people/places/taxa. This motivates the curated-seed +
   domain-filter pipeline (`scripts/term_memory/build_domain_glossary.py`).

## Framework validation — ACL-tagged term-recall (CORRECT harness) ✅

`eval/streaming_sst/score_acl_tagged.py` through the rasst-demo framework, with
the **lm-aware speech form** (agent fix: chunk = 0.96·lm s, retriever lookback
1.92 s → varctx encode window) and **lock-step feeding** (one segment → wait for
its partial → next, so each increment is exactly one chunk). Gold is non-circular:
per-sentence tagged terms from the glossary's `sentence_indices` (+ zh). Preset
`acl_tagged_gs10k` = `acl6060_tagged_gt_union_gs10000` (238 GT + 9762 wiki fillers).

**lm=2, 40 tagged sentences, 105 gold term-occurrences → TERM_RECALL = 0.8667 (91/105).**

This sits in the validated reference range (InfiniSST medicine hardraw lm1–4 =
0.79/0.83/0.85/0.85), confirming the framework's retrieval + term injection work.
Recovers e.g. `Annotated Corpus→注释语料库`, `machine translation→机器翻译`,
`NLP→自然语言处理`, `CRF→CRF`, `OOV→OOV`.

### Scale the glossary at lm=2 (controlled: same 238 GT + gold, only distractors grow) ✅

Same correct harness, same 40 sentences / 105 gold occurrences. The `acl_tagged_gs10k`
glossary (238 GT-with-`sentence_indices` + 9 762 wiki fillers) is padded with general
Wikidata distractors to 100k and 500k — the GT entries and their `sentence_indices`
are byte-identical across scales, so the gold denominator is held fixed and the only
variable is distractor count (500k ⊃ 100k, seed 1215).

Both recall **and retrieval precision are measured in this same harness** (precision
= fraction of injected refs whose key is one of the 238 curated GT terms — i.e. a
domain-relevant ref rather than a distractor; refs come from each partial's
`meta.references`):

| preset | terms | distractors | TERM_RECALL | RETRIEVAL_PRECISION | relevant refs | refs/chunk |
|---|---:|---:|---:|---:|---:|---:|
| `acl_tagged_raw` (GT-only) | 238       | 0       | 0.8286 (87/105) | **1.0000** (212/212)  | 212 | 1.37 |
| `acl_tagged_gs10k`  | 10,000    | 9,762   | **0.8762** (92/105) | 0.5134 (210/409)  | 210 | 2.64 |
| `acl_tagged_gs100k` | 100,000   | 99,762  | 0.8476 (89/105) | 0.2915 (207/710)  | 207 | 4.58 |
| `acl_tagged_gs500k` | 500,000   | 499,762 | 0.8667 (91/105) | 0.1633 (193/1182) | 193 | 7.58 |
| `acl_tagged_gs1m`   | 1,000,000 | 999,762 | 0.8190 (86/105) | 0.1414 (189/1337) | 189 | 8.52 |

(`acl_tagged_raw` = the curated 238 GT with zero distractor padding; by construction
every injected ref is a GT term → precision 1.0.)

**Recall holds within a tight band; precision collapses.** Across 0 → 1M distractors
term-recall stays in **0.819–0.876** (a ~6-occurrence spread on 105) with no strong trend —
mostly decoding noise, with only a slight tilt down at 1M (0.819, the low end, but the
gold is still retrieved — see below). Per-sentence diff: only a few sentences flicker
(#2 `NLP task`, #17 `vocabulary`, #28 `previous/newswire`), all common/boundary words; the
core domain terms 注释语料库, 机器翻译, CRF, OOV, 解析, 新闻专线 are recovered at every
scale. Notably the zero-distractor baseline is *not* the best on recall (0.8286) and does
not beat gs10k. Retrieval precision, by contrast, falls monotonically **1.00 → 0.51 →
0.29 → 0.16 → 0.14**.

The mechanism is explicit in the counts: the number of *relevant* (curated-GT) refs the
retriever surfaces is nearly flat across all five scales (**212 / 210 / 207 / 193 / 189**)
— the gold keeps being retrieved no matter how big the memory (it erodes only marginally,
a few terms slipping below top-k at 500k+) — while the total injected refs balloon
212 → 409 → 710 → 1182 → 1337 (refs/chunk 1.37 → 8.52) as distractors crowd in. Note the
noise *saturates*: 500k → 1M only raises refs/chunk 7.58 → 8.52 because the per-chunk
top-k caps injection, so precision barely moves (0.16 → 0.14). So the recall differences
are essentially noise (near-identical relevant injection), and a 500k–1M memory still
finds the gold but injects mostly noise into the prompt. → Keep the *active* slice ~10k
(high precision); a large background memory is safe for recall but wasteful (and subtly
risky) to inject wholesale.

> These precision numbers are measured in the correct lm-aware lock-step harness. They
> supersede the 0.49 → 0.14 → 0.08 from the earlier `score_terms.py` scale sweep (broken
> speech form); the trend matches, the exact values differ.

_Result JSONs: `runtime/term_memory/acl_tagged_lm2_gs{0,10,100,500}k_prec.json`,
`acl_tagged_lm2_gs1m_prec.json`._

> ⚠️ The `score_terms.py` numbers in the sections below (recall ~0.4) came from a
> **broken harness**: it streamed whole variable-length utterance wavs at the
> wrong rate (segmentation confound) with incidental zh-substring matching — wrong
> speech form *and* metric. They are NOT comparable to TERM_ACC and should be read
> only for the artifact-resistant ratio trends (retrieval precision vs scale).
> The number above (0.8667) is the trustworthy framework check.

## Terminology accuracy — ⚠️ first attempt was CIRCULAR (retracted)

A first pass set gold = the ACL-238 glossary's own terms appearing in the talk,
then measured term-recall per preset. **Invalid for cross-glossary comparison:**
`acl_tagged_raw` scored 1.000 only because it *is* the gold source. A coverage
check shows **0 / 15 of those gold terms exist in open_wiki_academic (19k) or
nlp_ai_cs (10k)** — so the open memories were penalized for not containing
ACL-238's literal strings, not for poor terminology (they retrieve real NLP terms
like `Natural language`, `Speech corpus` live). ACL-238 also contains many
non-terms (`words`, `English`, `language`, `compared`, `Online`) the base model
already translates, inflating its apparent edge. Measured the same way, *any*
glossary scores ~1.0 against its own terms — the metric, not the glossary, is the
problem.

**Correct approach:** a glossary-INDEPENDENT gold (the talk's genuine domain
terms with accepted translation variants — `eval/streaming_sst/acl268_gold_terms.json`),
measured across all presets and reported next to each glossary's *coverage* of
that gold (`score_terms.py --gold-file ... --coverage ...`).

### Fair re-eval (independent gold, 9 terms, talk 2022.acl-long.268)

| preset | gold coverage | term-recall | recovered |
|---|---:|---:|---|
| none | — | 0.778 (7/9) | lexical borrowing, annotated corpus, corpus, dataset, NLP, borrowing, model |
| acl_tagged_raw (238) | 5/9 | 0.778 (7/9) | linguistic borrowing, annotated corpus, corpus, dataset, NLP, borrowing, model |
| open_wiki_nlp_ai_cs (10k) | 2/9 | 0.444 (4/9) | dataset, NLP, borrowing, model |
| open_wiki_academic (19k) | 3/9 | 0.444 (4/9) | dataset, NLP, borrowing, model |

- **No glossary covers the talk's core terms** (`lexical borrowing`, `code
  switching`, `linguistic borrowing` have ~0 coverage everywhere), and the **base
  model already translates most of them** (none = 7/9). So this talk has little
  for terminology retrieval to fix.
- The curated ACL glossary **ties** the no-glossary baseline; the **broad open
  memories score lower** (4/9), dropping terms none got right — noisy retrieval
  distracted the model. → broad open memory needs higher precision (threshold /
  domain filter) to avoid hurting on common-term inputs.
- ⚠️ **Poor testbed.** Terminology RAG's value is on *rare* domain terms the base
  model gets *wrong*; this common-term talk doesn't have them. The experiment
  below uses the full ACL set with a fixed technical-term gold instead.

## Glossary-scale experiment (fixed gold denominator) — the right design

Method (per the fixed-gold, scale-the-glossary design): keep the full ACL-238
raw glossary in every tested inventory, and report terminology accuracy with
two denominators. `term_ACC_raw238` uses the full raw annotation so the paper
does not hide daily/common/generic terms; `term_ACC_technical142` uses
`eval/streaming_sst/acl_gold_technical.json`, the curated technical subset used
by the historical scale rows below. Build glossaries that contain the raw ACL
terms, padded with general-Wikidata distractors to 10k / 100k / 500k
(`scale_*` presets). So coverage is held ~constant and the only variable is
distractor count. Metrics should include both raw238 and technical142 versions
of term accuracy, `gold_retrieved`, and retrieval precision. Eval audio =
ACL-60-60.

**Validation — terminology retrieval works (full 468-seg set, all 142 gold spoken):**

| preset | recall@142 | gold_retrieved | retrieval_precision |
|---|---:|---:|---:|
| none | 0.789 | — | — |
| acl_tagged_raw (238) | **0.944** | 0.965 | 0.49 |

→ With a real technical-term gold and full audio, curated terminology lifts
recall **0.789 → 0.944**. (The earlier 9-term smoke talk had no headroom.)

**Scale curve (120-seg subset, real-time feed so segmentation is identical across
presets — verified: chunks 377–393 for all):**

| glossary (gold always present) | recall@142 | gold_retrieved | retrieval_precision | refs/chunk |
|---|---:|---:|---:|---:|
| none | 0.366 | — | — | 0.0 |
| acl-238 (0 distractors) | 0.458 | 0.465 | 0.486 | 1.7 |
| + 10k | **0.472** | 0.451 | 0.42 | 2.1 |
| + 100k | 0.437 | 0.458 | 0.14 | 5.6 |
| + 500k | 0.430 | 0.437 | 0.08 | 8.7 |

**Findings (answers "how large before it hurts"):**
- **Recall is robust to scale.** Even at 500k, recall@142 = 0.430 — statistically
  tied with the 0-distractor 238 (0.458) and far above none (0.366). The gold
  terms keep being retrieved (`gold_retrieved` ≈ 0.44 even at 500k) and the model
  picks them out of the distractor noise. You can scale the background memory to
  500k without losing recall.
- **Precision is the limiting factor** — it falls off a cliff between 10k and
  100k: 0.49 → 0.42 → **0.14** → 0.08, with refs/chunk ballooning 1.7 → 8.7. At
  100k+, the injected term_map is mostly noise (prompt bloat + subtle-error risk),
  even though recall survives.
- **Sweet spot ≈ 10k** for the active slice (recall peak 0.472, precision still
  0.42). → Architecture: keep the *active* slice small/high-precision (~10k,
  domain-adaptive); a large background memory is safe for recall but wasteful to
  inject wholesale. Raising the retrieval score threshold / top-k discipline is
  the lever to scale further without precision collapse.
- ⚠️ 120-seg subset (56/142 gold spoken) compresses absolute recall; the curve is
  for *relative* scale comparison. Distractors are general Wikidata; domain-
  relevant distractors would stress precision harder.

## Concurrency (continuous batching, open_wiki_academic 19k)

`eval/streaming_sst/concurrency_term_memory.py`: N clients stream the same audio
simultaneously against one shared term memory + retriever. Per-chunk latency
excludes each session's cold chunk.

| N | total chunks | wall (s) | throughput (seg/s) | retrieve p50 / p95 (ms) | gen p50 / p95 (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 88.0 | 0.17 | 60 / 78 | 420 / 642 |
| 8 | 120 | 88.7 | 1.35 | 71 / 154 | 482 / 1064 |
| 16 | 240 | 89.3 | 2.69 | 87 / 227 | 521 / 1130 |
| 32 | 460 | 89.6 | 5.13 | 162 / 394 | 665 / 1205 |

**Wall-clock stays ~88–90 s constant from N=1→32 while throughput scales ~30×.**
Per-chunk retrieve (60→162 ms p50) and generation (420→665 ms p50) degrade
sub-linearly and stay within the 1.92 s segment budget at 32 concurrent streams —
the coalescing micro-batch scheduler + vLLM continuous batching absorb the load.
(Feed is paced at 0.45 s/chunk, so wall-clock is feed-bound; the point is the
server keeps up without latency blow-up.)

## Not covered here (next)

BLEU / StreamLAAL on the full ACL 60-60 dev set (SimulEval harness; needs a
framework SimulEval agent). Concurrency (N=8/16/32) sweep under continuous
batching. Multilingual (ja/de) open-memory indexes (model is currently zh).

_Generated 2026-06-18 from `runtime/term_memory/sweep_results.json` + the 1M warm probe._

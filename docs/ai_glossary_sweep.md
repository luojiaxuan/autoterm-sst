# AI Glossary Size Sweep (2026-07-06)

This note records the full long-audio streaming benchmark for the AI glossary
size question. The run keeps the ACL raw glossary in every tested inventory and
adds sampled AI/RDF candidates at larger scales. Metrics are fixed across rows:
BLEU, `term_ACC` (`term_recall` in the harness), and `masked_term_BLEU`.

## Source Of Truth And Artifact Status

- Code repo: `git@github.com:luojiaxuan/rasst-demo.git`, branch `framework`.
- Run root: `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/strict_longaudio_sweep_20260706T110618Z`.
- Rows JSON: `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/strict_longaudio_sweep_20260706T110618Z/json_ws/strict_streaming_longaudio_rows.json`.
- Summary table: `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/strict_longaudio_sweep_20260706T110618Z/json_ws/strict_streaming_longaudio_summary.md`.
- Aries manifest used by the server: `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/manifest_ai_glossary_sweep_20260706_aries.json`.
- Server log: `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/server_logs/server_8012_tmpfix.log`.
- Status: local staging only. These generated glossaries, indexes, logs, and
  benchmark rows are not yet uploaded to Hugging Face. If they become reusable
  artifacts, publish them as a HF dataset and record the repo/revision here.

## AI Term Mining Scale

The RDF mining pass shows two different scales, depending on how strict the
filter is:

| source | rows scanned | candidate terms | note |
| --- | ---: | ---: | --- |
| Wikidata P31 RDF broad filter | 9728548 | 983090 | dominated by scholarly-article title matches; useful as an upper-bound/noise pool |
| balanced P31 description filter | 4524378 | 3575 | stricter AI/CS/NLP description matches |
| translated Wiki + existing AI/NLP/CS seed | n/a | 13580 | clean zh-usable local AI-ish pool after filtering |
| ACL raw glossary | n/a | 238 | always included in every sweep inventory |

Important limitation: the local clean Chinese AI glossary pool is only about
13.6k terms. The 50k and 100k rows below therefore use broad RDF candidates with
identity fallback. Treat them as a broad/noisy stress test, not as evidence that
we have a clean 100k zh AI glossary locally.

## Built Sweep Inventories

| artifact | terms | extra source | identity fallback | local path |
| --- | ---: | --- | ---: | --- |
| `acl_ai_translated_plus10k` | 10238 | wikidata_ai_translated:10000 | 0 | `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/glossaries/acl_ai_translated_plus10k.json` |
| `acl_ai_broad_plus10k` | 10238 | wikidata_ai_broad_scholarly:9170, wikidata_ai_broad_keyword:830 | 10000 | `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/glossaries/acl_ai_broad_plus10k.json` |
| `acl_ai_broad_plus50k` | 50238 | wikidata_ai_broad_scholarly:45924, wikidata_ai_broad_keyword:4076 | 50000 | `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/glossaries/acl_ai_broad_plus50k.json` |
| `acl_ai_broad_plus100k` | 100238 | wikidata_ai_broad_scholarly:91836, wikidata_ai_broad_keyword:8164 | 100000 | `/mnt/data1/jiaxuanluo/rasst_eval/ai_glossary_sweep/glossaries/acl_ai_broad_plus100k.json` |

## Streaming Evaluation Setup

- Host: Aries A6000, server port `8012`, Qwen3-Omni via in-process vLLM with MaxSim RAG.
- Audio: five ACL 60-60 WAVs concatenated, 3441.718 seconds total.
- Client streaming feed: `PACKET_SAMPLES=8000`, `FEED_SLEEP=0.45`. These are transport packets, not the server inference chunk size. The completed rows show lm=2 produced about 1685 partial chunks and lm=1 about 3180 partial chunks, so the server-side latency multiplier changed the processing stride as intended.
- Presets: `acl_tagged_raw`, `acl_ai_translated_plus10k`, `acl_ai_broad_plus10k`,
  `acl_ai_broad_plus50k`, `acl_ai_broad_plus100k`.
- Latency multipliers: `2` and `1`.
- The manual-preset sweep logs references surfaced under the fixed top-10 cap;
  `refs/chunk` can be below 10 because chunks without confident retrieved terms
  do not force filler refs in this harness. This benchmark is a scale/relevance
  sweep, not the unit test for the auto mode's exact-10 backfill invariant.

## Results: lm=2

| lm | setting | terms | term_ACC | BLEU | masked_term_BLEU | PromptGold@10 | RetrPrec@10 | refs/chunk | retr p50 ms | retr p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | acl_tagged_raw | 238 | 0.9720 | 58.27 | 53.10 | 0.9720 | 0.4800 | 1.5340 | 93.46 | 122.64 |
| 2 | acl_ai_translated_plus10k | 10238 | 0.9650 | 58.72 | 53.35 | 0.9720 | 0.2340 | 3.0870 | 95.39 | 118.21 |
| 2 | acl_ai_broad_plus10k | 10238 | 0.9720 | 58.38 | 52.75 | 0.9790 | 0.3670 | 1.9950 | 96.81 | 117.99 |
| 2 | acl_ai_broad_plus50k | 50238 | 0.9790 | 58.65 | 53.24 | 0.9790 | 0.1960 | 3.6400 | 94.47 | 121.90 |
| 2 | acl_ai_broad_plus100k | 100238 | 0.9650 | 58.83 | 53.32 | 0.9790 | 0.1370 | 5.0270 | 93.09 | 118.49 |

## Results: lm=1

| lm | setting | terms | term_ACC | BLEU | masked_term_BLEU | PromptGold@10 | RetrPrec@10 | refs/chunk | retr p50 ms | retr p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | acl_tagged_raw | 238 | 0.9440 | 54.93 | 49.43 | 0.9790 | 0.4790 | 1.0080 | 90.74 | 111.01 |
| 1 | acl_ai_translated_plus10k | 10238 | 0.9580 | 54.47 | 49.23 | 0.9790 | 0.2340 | 2.0530 | 90.67 | 110.31 |
| 1 | acl_ai_broad_plus10k | 10238 | 0.9510 | 55.36 | 49.82 | 0.9790 | 0.3710 | 1.2950 | 91.96 | 106.66 |
| 1 | acl_ai_broad_plus50k | 50238 | 0.9580 | 55.22 | 49.97 | 0.9720 | 0.1930 | 2.4370 | 90.38 | 114.14 |
| 1 | acl_ai_broad_plus100k | 100238 | 0.9580 | 54.85 | 49.70 | 0.9510 | 0.1330 | 3.4740 | 97.77 | 115.88 |

## Surfaced-Term Diagnostics

The conditional term metrics separate output correctness when a gold term was surfaced by retrieval from output correctness when it was not surfaced. This is why flat `term_ACC` should not be read as proof that the prompt channel stayed healthy.

| lm | setting | chunks | term_ACC surfaced | term_ACC not surfaced |
| --- | --- | ---: | ---: | ---: |
| 2 | acl_tagged_raw | 1681 | 0.978 | 0.75 |
| 2 | acl_ai_translated_plus10k | 1684 | 0.978 | 0.5 |
| 2 | acl_ai_broad_plus10k | 1689 | 0.978 | 0.667 |
| 2 | acl_ai_broad_plus50k | 1685 | 0.986 | 0.667 |
| 2 | acl_ai_broad_plus100k | 1685 | 0.971 | 0.667 |
| 1 | acl_tagged_raw | 3190 | 0.964 | 0.0 |
| 1 | acl_ai_translated_plus10k | 3202 | 0.978 | 0.0 |
| 1 | acl_ai_broad_plus10k | 3182 | 0.95 | 1.0 |
| 1 | acl_ai_broad_plus50k | 3185 | 0.971 | 0.5 |
| 1 | acl_ai_broad_plus100k | 3174 | 0.978 | 0.571 |

## Interpretation

For the requested question, there is no clear translation-quality regression up
to the tested 100k broad RDF inventory when ACL raw terms are always included.
`term_ACC`, BLEU, and `masked_term_BLEU` stay in the same band for both latency
settings:

- lm=2: `term_ACC` ranges 0.965-0.979; `masked_term_BLEU` ranges 52.7533-53.3536.
- lm=1: `term_ACC` ranges 0.944-0.958; `masked_term_BLEU` ranges 49.2346-49.9715.

The systematic regression is retrieval relevance, not output quality. Retrieval
precision drops as broad inventory size grows:

- lm=2 precision: 0.480 oracle -> 0.367 broad+10k -> 0.196 broad+50k -> 0.137 broad+100k.
- lm=1 precision: 0.479 oracle -> 0.371 broad+10k -> 0.193 broad+50k -> 0.133 broad+100k.

At 100k, `PromptGold@10` is still high for lm=2 (0.979) but drops to 0.951 for
lm=1. That is the first visible retrieval-channel warning. It does not yet
translate into a large BLEU/masked-BLEU drop in this run, but it means the
prompt evidence is mostly distractors and the method is relying more on model
knowledge and ranking luck.

## Route Policy After This Run

The automatic glossary route should be domain-specific and should not prepend a
common glossary. The production route policy is:

1. Start from a domain-specific initial slice, currently `nlp_core_10k` for the
   ACL/AI demo.
2. Route among domain slices using speech-embedding centroid similarity plus
   retrieved-reference metadata; do not use source transcripts, generated text,
   or manual terms to infer the domain. A slice change now also requires the
   target to beat the current active slice, survive consecutive candidate
   windows, and pass a post-switch cooldown.
3. Keep the prompt candidate budget fixed at 10. Inventory size changes the pool
   from which candidates are ranked, not the prompt interface.
4. Use clean domain slices as the default active inventory. The available clean
   zh AI pool is around 13.6k, so a 10k AI/NLP core slice is the correct default
   until a larger clean AI glossary exists.
5. Treat 50k/100k broad RDF pools as rescue/diagnostic inventories rather than
   default routed domains. They did not cause a clear BLEU or term_ACC collapse,
   but precision falls below 0.20 at 50k and near 0.13 at 100k.
6. Add diagnostics around retrieval precision, `PromptGold@10`, refs/chunk, and
   `TermRecall | surfaced` vs `TermRecall | not surfaced`; output term accuracy
   alone hides prompt-channel failure.

Concrete threshold from this run: quality metrics do not show a clear rollback
through 100k, but retrieval relevance visibly degrades by 50k and is poor by
100k. Therefore the default auto route should stay on clean domain-core slices,
with broad 50k/100k inventories only used as fallback candidate pools or
upper-bound diagnostics.

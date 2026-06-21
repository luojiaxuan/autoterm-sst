# RASST-Demo Paper Artifacts, 2026-06-21

Runtime: Taurus GPU service job `46544`, Qwen3-Omni vLLM TP=2 on GPUs
`6,7`, MaxSim retriever on GPU `5`, framework port `8011`.

Figure 2 screenshot:

```text
runtime/eval_20260621/figure2_ui.png
```

The screenshot uses the current web UI plus live Taurus JSON WebSocket output
from `auto_working` on the ACL6060 smoke audio. It shows translated text,
retrieved terms, active glossary state, active term count, and retrieve/generate
latency.

## Table 1: Quality / User-Effort Comparison

Scope: ACL6060 smoke talk `2022.acl-long.268`, first 8 segments, independent
9-term gold file `eval/streaming_sst/acl268_gold_terms.json`. `masked_bleu`
is `masked_terms_bleu` from `score_terms.py`; retrieval p95 and active terms
come from the framework JSON WebSocket eval. `StreamLAAL` is left pending because
this JSON-WS harness does not produce SimulEval token-delay logs.

| setting | user effort | active terms | term acc | masked_bleu | StreamLAAL | retrieve p95 |
|---|---:|---:|---:|---:|---:|---:|
| none | 0 | 0 | 0.444 | 42.34 | pending | -- |
| broad open | 0 | 100,000 | 0.444 | 45.29 | pending | 111.54 ms |
| auto working | 0 | 10,000 | **0.889** | 45.80 | pending | 88.20 ms |
| curated oracle | high | 238 | 0.778 | **47.07** | pending | 81.13 ms |

Takeaway for reviewer framing: on this smoke slice, zero-setup `auto_working`
improves term accuracy over both no glossary and broad open memory, while
retrieving fewer refs/chunk and keeping p95 retrieval latency below broad open.
The curated oracle still has the best masked BLEU and lowest RAG latency, but it
requires manual/curated terminology effort.

Primary sources:

```text
runtime/eval_20260621/table1_auto_glossary.json
runtime/eval_20260621/table1_terms_smoke.json
runtime/eval_20260621/table1_auto_glossary.md
```

## Table 2: System Scalability / Retrieval Latency

Scope: warm steady-state MaxSim retrieval over the framework JSON WebSocket.
Cold-load values are first-activation index load measurements from the complete
238 -> 1M scale sweep; the current `nlp_core_10k` quick sweep was already
preloaded, so its cold load is reported from the earlier true-cold 10k scale
measurement.

| memory | terms | retrieve p50/p95 | refs/chunk | cold load |
|---|---:|---:|---:|---:|
| curated | 238 | 61.3 / 105.3 ms | 2.33 | ~0.5 s |
| domain | 10,000 | 78.5 / 82.1 ms | 1.80 | ~0.3 s |
| open | 100,000 | 64.1 / 82.6 ms | 4.89 | ~6.8 s |
| stress | 1,000,000 | 78 / 95 ms | 9.62 | >30 s |

Takeaway for reviewer framing: warm retrieval latency is stable through 1M
terms; the scaling cost is cold loading and reference noise, not per-chunk
MaxSim compute. This supports keeping the active runtime glossary compact while
maintaining a larger offline/open memory.

Primary sources:

```text
/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/sweep_results.json
/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/acl_tagged_lm2_gs1m_prec.json
runtime/eval_20260621/table2_nlp_core_10k_sweep.json
docs/open_term_memory_eval.md
```

## Remaining Gap

For a camera-ready Table 1, rerun a preset-aware SimulEval harness to fill the
`StreamLAAL` column for `none`, `broad open`, `auto working`, and `curated
oracle`. The current table is valid for terminology quality and retrieval
latency, but it should not pretend that JSON WebSocket timing is canonical
StreamLAAL.

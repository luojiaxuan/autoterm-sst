# AutoTerm-SST Paper Artifacts, 2026-07-09

Runtime: Taurus GPU service job `46544`, Qwen3-Omni vLLM TP=2 on GPUs
`6,7`, MaxSim retriever on GPU `5`, framework port `8011`.

Figure 2 screenshot:

```text
runtime/eval_20260621/figure2_ui.png
```

The screenshot uses the AutoTerm-SST web UI plus a captured diagnostic
`common_10k` session. It illustrates translated text, retrieved terms, active
glossary state, active term count, and retrieve/generate latency; the released
automatic path starts from domain-specific slices.

## Table 1: Current ten-talk mixed-domain comparison

The paper's main table uses the full ten-talk benchmark-aligned union
evaluation: five ACL and five medicine talks streamed alternately through one
session. The technical-gold score covers 875 occurrences (193 ACL and 682
medicine). The source-of-truth artifact is staged on Aries at
`/mnt/data3/jiaxuanluo/eval_out/10talk_zh/term_acc_10talk.json`; it is a
local-only evaluation artifact and has not been uploaded to Hugging Face. The
Git-tracked result summary and provenance are in
`docs/auto_glossary_mixed_switch_20260707.md`.

| setting | session setup | combined term acc. | ACL | medicine | BLEU |
|---|---|---:|---:|---:|---:|
| no memory | none | 0.744 | 0.782 | 0.733 | 58.84 |
| fixed NLP | select NLP | 0.744 | 0.907 | 0.698 | 59.18 |
| fixed medicine | select medicine | 0.901 | 0.772 | 0.937 | **59.63** |
| Automatic | none | **0.936** | **0.912** | **0.943** | 58.10 |

### Earlier truncated three-talk comparison (provenance only)

The earlier ACL$\rightarrow$medicine$\rightarrow$ACL result covered 143
technical-gold occurrences and produced 0.916 combined term accuracy for
Automatic, versus 0.839 for fixed NLP and 0.825 for fixed medicine. Its local
staging artifacts are
`/mnt/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260708_union_truncated_8013/term_acc_compare_v2.json`
and `auto_working_v2.json`. The paper retains this result only as the explicitly
labelled truncated three-block probe in the appendix.

## Legacy smoke artifact: historical quality / user-effort comparison

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

This historical table is retained only for provenance. Its `auto_working` row
uses the old `common_10k`/general router and must not be presented as evidence
for the released domain-slice `hybrid_window_topic` configuration.

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

For a camera-ready latency table, rerun a preset-aware SimulEval harness to fill the
`StreamLAAL` column for `none`, `broad open`, `auto working`, and `curated
oracle`. The current table is valid for terminology quality and retrieval
latency, but it should not pretend that JSON WebSocket timing is canonical
StreamLAAL.

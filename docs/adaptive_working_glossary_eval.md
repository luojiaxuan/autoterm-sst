# Adaptive Working Glossary Evaluation

This eval isolates the terminology-memory behavior of the zero-setup adaptive
glossary. It should be run after the working slice manifest points at real
MaxSim indexes and router centroids under:

```text
/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/
```

## Conditions

Run at least:

| condition | preset |
|---|---|
| no terminology | `none` |
| curated oracle | `acl_tagged_raw` |
| wrong manual domain | `medicine_core_10k` on ACL/NLP audio |
| correct manual domain | `nlp_core_10k` on ACL/NLP audio |
| zero-setup adaptive | `auto_working` |
| large open-memory stress | `open_wiki_100k` or `open_wiki_500k` |

## Commands

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo

python scripts/term_memory/build_domain_centroids.py \
  --manifest /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/manifests/current.json \
  --out-dir /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/centroids \
  --presets nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k \
  --target-lang zh \
  --update-manifest

python eval/streaming_sst/eval_auto_glossary_switch.py \
  --acl-text /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/source_text.txt \
  --medicine-text /mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh/medicine.source_text.en__medicine_404.txt \
  --max-windows-per-domain 8 \
  --max-switch-windows 4 \
  --out-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_switch_router_only_20260707.json

python eval/streaming_sst/eval_auto_glossary.py \
  --base-url http://127.0.0.1:8011 \
  --seg-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg \
  --presets none,acl_tagged_raw,medicine_core_10k,nlp_core_10k,auto_working,open_wiki_100k \
  --language-pair "English -> Chinese" \
  --out-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary.json

python eval/streaming_sst/score_terms.py \
  --base-url http://127.0.0.1:8011 \
  --seg-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg \
  --source-text /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/source_text.txt \
  --reference-text /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/ref.txt \
  --gold-file eval/streaming_sst/acl_gold_technical.json \
  --mask-glossary /mnt/taurus/data2/jiaxuanluo/RASST/data/glossaries/acl6060_tagged_gt_raw_min_norm2.json \
  --sacrebleu-tokenizer zh \
  --presets none,acl_tagged_raw,medicine_core_10k,nlp_core_10k,auto_working,open_wiki_100k \
  --out-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_terms.json

python eval/streaming_sst/score_auto_glossary.py \
  --auto-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary.json \
  --term-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_terms.json \
  --out-md /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_table.md
```

Taurus source-text sanity run on 2026-07-07 was rerun at
`/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_switch_router_only_20260707_final8.json`
and passed all four router-unit scenarios. ACL-only and medicine-only had zero
false switches. ACL->medicine passed within the 4-window threshold; the clean
fixture regression with contested synthetic probe evidence passed the stricter
2-window text-path threshold at
`/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_switch_fixture_probe_20260707_final8.json`.
The audio-only probe fallback is separately covered by unit tests that require
raw probe top-score, raw margin, and at least two positive probe domains before
the renormalized probe score can switch domains.

`eval_auto_glossary_switch.py` is a router-unit/source-text diagnostic. It
directly drives `HybridWindowTopicRouter`, disables wall-clock update/cooldown
delays, and uses synthetic probe scores when `--with-probe` is set. These
numbers validate the window-topic state machine and provide a source-text upper
bound; they are not an end-to-end Omni/MaxSim deployment-latency benchmark.
The deployable E2E path uses speech-window domain probes plus delayed generated
target-translation text, not source transcripts or ASR text. That path is wired
in the runtime, but full generated-target E2E switch-quality benchmarking is
still pending.

Mixed ACL/medicine switch benchmark was added on 2026-07-07 in
`docs/auto_glossary_mixed_switch_20260707.md`. It uses ACL 5 talks and medicine
5 speeches, with target/reference text windows as generated-target proxies and
controlled clean expected domain-probe evidence. Fixed 64 windows/item passed
both alternating and random playlists: alternating had 9/9 transitions within
3 windows, random seed 20260707 had 7/7 transitions within 3 windows, both with
steady-state domain accuracy 1.0 and zero wrong switches. The no-probe,
inverted-probe, and contested-probe diagnostics failed as expected, confirming
that this benchmark can expose missing, wrong, or weak probe evidence. It is a
state-machine/proxy validation, not proof of real MaxSim speech-probe domain
discrimination.

`eval/streaming_sst/eval_mixed_audio_switch.py` is the next harness for the real
deployment path. It streams ACL/medicine audio blocks through one JSON WS
session, records `meta.domain_probe_scores`, generated-target router text
metadata, active glossary preset/domain, prompt candidate counts, switch
latency, and steady-state domain accuracy against playlist spans. A dry-run can
verify the 5+5 audio playlist without a server:

The live-run schema is tied to the current runtime emitter in
`framework/agents/omni.py::_event_meta`, not the older high-level docs. The
harness fails hard if required runtime fields such as `cursor_samples`,
`domain_probe_scores`, `topic`, `topic_router`, `router_text_source`,
`fixed_prompt_k`, or `candidate_pool_count` are missing, because span-aligned
metrics are meaningless without them.

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo

python3 eval/streaming_sst/eval_mixed_audio_switch.py \
  --schedule alternating \
  --dry-run \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_dryrun/alternating_audio_playlist.json
```

Dry-run on Taurus at Git ref `441a15d` succeeded under
`/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_441a15d/`.
The alternating and random playlists both contain 10 blocks and 16,848.115
seconds of audio: 5 ACL talks and 5 medicine speeches. This is about 4.68 hours
of audio before model generation overhead, so full E2E should be treated as a
long run rather than a smoke test.

A 20-second schema smoke against the old Taurus `127.0.0.1:8011` server reached
a partial event but failed fast because that server's partial metadata did not
include `domain_probe_scores` or `router_text_source`. This was expected for the
invalid server state (`router_mode=embedding_refs`) and confirmed the harness
does not fabricate span-aligned metrics when required routing metadata is absent.

The valid E2E server is now Taurus `127.0.0.1:8012`, launched with
`router_mode=hybrid_window_topic`, manifest
`auto_working_alias_20260619T204803Z`, and auto presets
`nlp_core_10k,medicine_core_10k`. Use 1.92s real-time feeding and a real
streaming switch tolerance:

```bash
python3 eval/streaming_sst/eval_mixed_audio_switch.py \
  --base-url http://127.0.0.1:8012 \
  --acl-items 1 \
  --medicine-items 1 \
  --schedule acl_then_medicine \
  --preset auto_working \
  --latency-multiplier 2 \
  --feed-sleep 1.92 \
  --max-switch-seconds 30 \
  --max-seconds-per-item 120 \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012/acl1_medicine1_120s_realtime_manifest_textfirst_8012.json \
  --out-md /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012/acl1_medicine1_120s_realtime_manifest_textfirst_8012.md
```

Current server/GPU status on 2026-07-07:

- Taurus `127.0.0.1:8012` is the valid E2E probe server. It uses physical GPUs
  4/5 for vLLM TP=2 and GPU6 for RAG/term retrieval. GPU7 is free, but not enough
  for another TP2+RAG server.
- Taurus GPUs 0-3 are occupied by another high-utilization job. Taurus data1 is
  the current output staging disk for eval artifacts.
- Aries has GPUs 4-7 free after cleanup, but `/` remains 100% full. The existing
  vLLM container cannot see CUDA, the GPU-visible containers lack vLLM/librosa,
  and host conda envs are missing or have CUDA/NCCL library mismatches. Do not
  start a production benchmark server there until a clean `/mnt/data3` venv or
  container is built.

Short real E2E results under
`/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012`:

| run | result |
|---|---|
| ACL-only 80s | active domain stayed `nlp` for all 41 events; wrong switches 0; retrieval p95 86.97ms |
| medicine-only 80s | switched from initial `nlp` to `medicine` at 59.52s; wrong switches 0 |
| ACL 120s -> medicine 120s | switched to `medicine` 20.16s after boundary; wrong switches 0; steady-state accuracy 1.0 with `--max-switch-seconds 30`; retrieval p95 88.66ms |

The full 5 ACL + 5 medicine real-time runs launched at commit `d63202d` were
canceled because the full playlist is longer than needed for the current router
question. The replacement run is a 4-block playlist under
`/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012_4block`:

```text
ACL 120s -> medicine 120s -> ACL 120s -> medicine 120s
```

It produced 3/3 correct switches, 0 wrong switches, retrieval p95 88.29ms, and
transition latencies of 20.16s, 17.28s, and
37.44s. The strict 30s tolerance fails only on the final medicine_606 switch;
40s/45s tolerance passes on the same record.

The more relevant 4-block terminology comparison is staged under
`/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_termacc_4block`.
It compares fixed `nlp_core_10k`, fixed `medicine_core_10k`, a composed
`manual_fixed_by_domain` row, and `auto_working` on the same 480s audio. The
mixed scorer is `eval/streaming_sst/score_mixed_audio_terms.py`; it uses ACL
source-filtered gold plus RASST hard medicine oracle term maps.

| denominator | run | term_acc | hits/gold | ACL acc | medicine acc |
|---|---|---:|---:|---:|---:|
| technical+medicine | fixed_nlp | 0.8043 | 37/46 | 0.7647 | 0.9167 |
| technical+medicine | fixed_medicine | 0.6957 | 32/46 | 0.7059 | 0.6667 |
| technical+medicine | manual_fixed_by_domain | 0.7391 | 34/46 | 0.7647 | 0.6667 |
| technical+medicine | auto_working | 0.7174 | 33/46 | 0.6471 | 0.9167 |
| raw+medicine | fixed_nlp | 0.7297 | 54/74 | 0.6935 | 0.9167 |
| raw+medicine | fixed_medicine | 0.6757 | 50/74 | 0.6774 | 0.6667 |
| raw+medicine | manual_fixed_by_domain | 0.6892 | 51/74 | 0.6935 | 0.6667 |
| raw+medicine | auto_working | 0.7162 | 53/74 | 0.6774 | 0.9167 |

Important caveat: this is output-centric term accuracy, not prompt-channel
attribution. Fixed `nlp_core_10k` still hits 11/12 medicine oracle occurrences,
so the base model can recover many medicine terms without the active medicine
slice. On this small sample, `auto_working` matches the best medicine accuracy
but loses ACL technical hits versus fixed `nlp_core_10k`.

The 480s run is now superseded for metric interpretation by the longer
ACL -> medicine_606 -> ACL real-time run in
`docs/auto_glossary_mixed_switch_20260707.md`, which uses 4142.4s of audio and
382 raw+medicine gold occurrences. It confirms the router switches correctly
using `generated_target` as `router_text_source` for 2081/2081 records, not
source transcript/ASR, but also shows the current broad `wiki_medicine` slice
has only 1/54 exact
coverage on `medicine_606` unique gold terms, so fixed NLP looking competitive
on medicine is mostly base-model recovery rather than useful glossary evidence.

## Remaining AutoTerm Todos

| status | item | note |
|---|---|---|
| done | Domain-specific `auto_working` without common base slice | Default route is among domain slices; common terms are diagnostic only, not a default prompt inventory. |
| done | Top-10 retrieval cap with score-filtered prompt refs | Runtime retrieves up to 10 candidates and does not backfill after filtering. |
| done | Target-text/probe state-machine proxy benchmark | ACL 5 + medicine 5 fixed-64 and full-window proxy runs passed under clean expected probe evidence. |
| done | Router guards for generic generated text, weak probe, and centroid-only false switches | Covered by unit tests. |
| done | Real mixed-audio harness | `eval_mixed_audio_switch.py` added and dry-run verified on Taurus. |
| done | Short real E2E generated-target switch probe | Taurus 8012 produced ACL-only, medicine-only, and ACL->medicine mixed results with zero wrong switches. |
| done | 4-block E2E generated-target switch benchmark | ACL -> medicine -> ACL -> medicine, 3/3 correct switches, 0 wrong switches; final switch latency 37.44s exceeds strict 30s tolerance. |
| done | 4-block mixed-domain term_ACC comparison | Fixed glossary vs `auto_working` table added for early diagnosis. |
| done | Long ACL -> medicine_606 -> ACL real-time comparison | Fixed NLP, fixed medicine, and `auto_working` evaluated with term_ACC, BLEU, and masked_term_BLEU. |
| in progress | Route threshold retuning from real probe failure modes | Current tuning uses generated-target text first and `current_margin_threshold=0.30`; speech probe is noisy and should stay auxiliary. |
| in progress | Medicine slice quality fix | Union-ready GT + builder landed 2026-07-08: `scripts/term_memory/build_gt_union_gs_glossary.py`, HF `glossaries/hard_medicine_gt_raw_unique212.json` (revision `204ba141`). Remaining: build `medicine_hardraw_gt_union_gs10000` with the gemini wiki filler, index it, register `medicine_hardraw_gs10k` + eval-only `medicine_hardraw_oracle`, rerun the 3-talk comparison. |
| pending | Paper claim update | Claim can mention long real E2E routing, but should not claim current `medicine_core_10k` improves medicine term_ACC until the slice-quality issue is fixed. |

## Metrics

| metric | source |
|---|---|
| `TERM_RECALL` | `score_terms.py` |
| false-copy rate | `score_terms.py` when available |
| regular BLEU | `score_terms.py --reference-text ...` |
| masked-term BLEU | `score_terms.py --reference-text ...`; removes target-side glossary translations from hyp/ref before sacreBLEU |
| reference precision | existing ACL tagged precision harness or postprocess refs vs gold |
| refs/chunk | `eval_auto_glossary.py` JSON metadata |
| surviving prompt refs/chunk | `prompt_reference_count` metadata after score filtering |
| prompt shortfall chunks | chunks where fewer than the top-10 retrieval cap survived filtering |
| retrieval p50/p95 | `retrieve_s` metadata |
| switch count | `meta.topic.switch_count` |
| router action/reason | `meta.topic_router.action` / `meta.topic_router.reason` |
| router-only switch success | `eval_auto_glossary_switch.py` |
| switch latency windows | `eval_auto_glossary_switch.py` |
| false medicine switch on ACL windows | `eval_auto_glossary_switch.py` |
| routing-only domain probe scores | `meta.domain_probe_scores` |
| time to first switch | first chunk whose switch count increases |
| active glossary over time | `meta.topic.active_glossary_preset` |

`score_terms.py` requires `sacrebleu` only when `--reference-text` is set. If
the active Python environment does not provide it, the JSON row records
`bleu_error` and the terminology/retrieval metrics still run.

## Desired Result

The target result is not that `auto_working` beats the curated oracle. The target
result is:

```text
auto_working requires zero user configuration
auto_working approaches correct-domain manual performance
auto_working beats or avoids wrong-domain manual behavior
auto_working injects fewer noisy refs than large open-memory presets
auto_working keeps retrieval latency within the streaming budget
```

If `auto_working` fails to switch on a short clip, lower
`RASST_AUTO_GLOSSARY_WARMUP_SEC` and `RASST_AUTO_GLOSSARY_UPDATE_SEC` for the
smoke test, then keep the production defaults at 30s/45s for the demo.

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

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo

python3 eval/streaming_sst/eval_mixed_audio_switch.py \
  --schedule alternating \
  --dry-run \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_dryrun/alternating_audio_playlist.json
```

Dry-run on Taurus at Git ref `b693168` succeeded under
`/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_b693168/`.
The alternating and random playlists both contain 10 blocks and 16,848.115
seconds of audio: 5 ACL talks and 5 medicine speeches. This is about 4.68 hours
of audio before model generation overhead, so full E2E should be treated as a
long run rather than a smoke test.

When the demo server is live on `127.0.0.1:8011`, run a short real replay first:

```bash
python3 eval/streaming_sst/eval_mixed_audio_switch.py \
  --base-url http://127.0.0.1:8011 \
  --schedule alternating \
  --preset auto_working \
  --latency-multiplier 2 \
  --max-seconds-per-item 60 \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_realprobe/alternating_60s_auto_working.json \
  --out-md /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_realprobe/alternating_60s_auto_working.md
```

Current server/GPU status on 2026-07-07:

- Taurus `127.0.0.1:8011` is healthy but not valid for this target eval:
  `router_mode=embedding_refs`, active term-memory snapshot is the AI glossary
  sweep, and it is not the `hybrid_window_topic` generated-target/probe path.
- Taurus preflight selected only GPU4 as free; the running server owns GPUs
  5/6/7 and other jobs occupy 0-3. This is not enough to start a separate TP2
  Omni server plus RAG device.
- Aries has `/` full and GPUs 0-5 already holding about 41-45GB each; only GPU7
  is empty and GPU6 has partial memory use. Do not start a new long server there
  without cleanup/coordination.

## Remaining AutoTerm Todos

| status | item | note |
|---|---|---|
| done | Domain-specific `auto_working` without common base slice | Default route is among domain slices; common terms are diagnostic/backfill only. |
| done | Fixed top-10 prompt candidate invariant | Covered by code/tests and previous benchmark docs. |
| done | Target-text/probe state-machine proxy benchmark | ACL 5 + medicine 5 fixed-64 and full-window proxy runs passed under clean expected probe evidence. |
| done | Router guards for generic generated text, weak probe, and centroid-only false switches | Covered by unit tests. |
| done | Real mixed-audio harness | `eval_mixed_audio_switch.py` added and dry-run verified on Taurus. |
| blocked | Real speech-window domain-probe replay result | Need a live server configured with `router_mode=hybrid_window_topic`; current Taurus server is `embedding_refs`. |
| pending | Full E2E generated-target switch benchmark | Need actual model outputs and generated-target router windows over mixed ACL/medicine audio. |
| pending | Mixed-domain BLEU / term_ACC / masked_term_BLEU | Need combined ACL+medicine references and medicine term gold/metric mapping. |
| pending | Route threshold retuning from real probe failure modes | Depends on real probe/E2E results: clean vs contested vs wrong probe behavior. |
| pending | Paper claim update | Current claim should stay limited to state-machine proxy until real E2E evidence exists. |

## Metrics

| metric | source |
|---|---|
| `TERM_RECALL` | `score_terms.py` |
| false-copy rate | `score_terms.py` when available |
| regular BLEU | `score_terms.py --reference-text ...` |
| masked-term BLEU | `score_terms.py --reference-text ...`; removes target-side glossary translations from hyp/ref before sacreBLEU |
| reference precision | existing ACL tagged precision harness or postprocess refs vs gold |
| refs/chunk | `eval_auto_glossary.py` JSON metadata |
| fixed prompt refs/chunk | `prompt_reference_count` metadata; invariant/debug only, not retrieval-quality evidence |
| prompt shortfall chunks | invariant/debug only; should be zero for fixed-budget auto mode |
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

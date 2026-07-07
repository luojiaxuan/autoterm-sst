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

Taurus source-text sanity run on 2026-07-07 passed with ACL->medicine latency 4
windows and medicine->ACL latency 2 windows. The fixture unit test keeps the
stricter two-window threshold for clean topic windows.

`eval_auto_glossary_switch.py` is a router-unit/source-text diagnostic. It
directly drives `HybridWindowTopicRouter`, disables wall-clock update/cooldown
delays, and uses synthetic probe scores when `--with-probe` is set. These
numbers validate the window-topic-first state machine; they are not an
end-to-end live-ASR/Omni/MaxSim probe deployment-latency benchmark.

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

# Domain-Specific Adaptive Working Glossary

RASST-Demo maintains a large open terminology memory offline, but activates a
fixed 10-candidate prompt list for each streaming session. The active inventory is
selected automatically from recent window topic evidence. The automatic path now
keeps a compact `common_terms` base slice active and switches one routed domain
overlay at a time.

## Architecture

```text
offline open memory
  -> working slices: nlp_core_10k / medicine_core_10k / finance_core_10k / legal_core_10k
  -> manifest: preset id -> terms.jsonl + maxsim index + domain metadata + centroid
  -> runtime session starts in auto_working
  -> common_terms stays active as the base retrieval slice
  -> active preset starts as the configured domain-specific overlay
  -> router observes recent source/ASR topic text, optional domain probes, speech embedding, and refs
  -> HybridWindowTopicRouter scores window topic first
  -> ActiveGlossaryManager preloads and atomically activates target index
  -> future chunks retrieve from common_terms + active domain overlay
  -> prompt receives top 10 refs, UI receives top 10 refs + router metadata
```

The framework boundary stays unchanged. `framework/app.py` and
`framework/router.py` still only move REST/WebSocket messages and
`TranslationEvent`s. All adaptive behavior lives inside `OmniAgent`, because
retrieval, prompting, batching, model state, and glossary selection are agent
concerns.

## Runtime Modules

- `framework/agents/term_memory/domain_taxonomy.py`: fallback/default domain
  labels, weighted high-precision topic keywords, and offline working-slice
  ranking helpers.
- `framework/agents/term_memory/topic_router.py`: `HybridWindowTopicRouter`
  routes from recent source/ASR topic text first, with domain-probe retrieval
  and speech-centroid signals as secondary evidence. The older
  `AudioNativeActiveGlossaryRouter` remains available as `embedding_refs`; the
  old keyword router remains only as `RASST_ROUTER_MODE=legacy_keywords`.
- `framework/agents/term_memory/active_glossary.py`: maps topic decisions to
  concrete active presets and handles fallback when a slice is unavailable.
- `framework/agents/plugins/retrieval.py`: adds `preload_index()`,
  `activate_index()`, `is_index_ready()`, and `retrieve_with_metadata()` so the
  agent can route from the same speech-side MaxSim pass used for term retrieval.
- `framework/agents/omni.py`: stores per-session adaptive state, schedules
  background switches after preload, caps prompt refs, and emits topic/router
  metadata.
- `serve/static/index.html`: shows mode, auto topic, confidence, active
  glossary, active terms, switch count, and router action from JSON WebSocket
  metadata or `/health`.

## Session State

Each `OmniSession` tracks:

```text
requested_glossary_preset   # user-facing mode, usually auto_working
active_glossary_preset      # concrete retrieval preset, e.g. nlp_core_10k
active_domain               # nlp / medicine / finance / legal
router_text_window          # source/ASR/topic text for routing, if available
router_text_source          # manifest_source / streaming_asr / generated_target / none
topic_confidence
last_topic_update_s
topic_history
glossary_switch_count
recent_references
router_state                 # EMA speech embedding, candidate streak, pending target
last_router_decision
```

When the user selects a concrete preset (`none`, `acl_tagged_raw`,
`nlp_core_10k`, etc.), the session becomes manual and topic routing is disabled.
When the user selects `auto_working`, topic routing is enabled and the initial
active preset is `RASST_AUTO_GLOSSARY_DEFAULT` (`nlp_core_10k` by default).

## Routing Logic

The default router is `RASST_ROUTER_MODE=hybrid_window_topic`. It is a
window-topic-first router: recent source-side text is the primary signal when
available. In controlled eval this text can come from RASST source segments; in
deployable live mode it should come from streaming ASR. Generated target text is
only a weak diagnostic signal because it can be biased by the current glossary
and model errors.

The router combines four signals:

```text
score(domain) =
    0.60 * text_topic_score
  + 0.25 * domain_probe_retrieval_score
  + 0.10 * speech_centroid_score
  + 0.05 * metadata_prior
```

`text_topic_score` uses high-precision weighted keywords from
`domain_taxonomy.py`. `domain_probe_retrieval_score` is a routing-only small
top-k probe over ready candidate domain indexes; fresh probe runs are gated by
the router warmup/update/cooldown schedule and use raw top-k scores rather than
the prompt retrieval score threshold. During the update/cooldown gate, the
router reuses cached probe scores so the consistency-window state machine still
sees stable probe evidence without rerunning MaxSim. When source/ASR topic text
is available, fresh probes are refreshed on the normal router update interval;
when no topic text is available, fresh probes refresh on the streaming window
cadence so audio-only domain changes are not frozen for the full update
interval. It must not change the prompt candidate budget.
`speech_centroid_score` is a weak tie-breaker based on offline domain centroids:

```text
centroid = normalize(mean(normalize(text_embs), dim=0))
```

`metadata_prior` uses metadata such as `active_glossary_preset`, `domain`, and
`source_preset`, but it is intentionally small so the current active slice does
not veto a high-confidence window-topic switch.

The earlier `embedding_refs` implementation is not sufficient for robust cross-domain switching
when a session moves from an ACL/NLP talk to a medicine talk. A 2026-07-07
probe found that real medicine audio windows often remained closer to the NLP
centroid than to the medicine centroid, while window-level source text/topic
signals cleanly identified the medicine domain. See
[`auto_glossary_routing_probe_20260707.md`](auto_glossary_routing_probe_20260707.md).

When no source/ASR/topic text is available, the fallback should use
small-top-k domain-probe retrieval plus weak centroid similarity. Current
active-slice metadata should be treated as a small prior, not as a veto against
a high-confidence text-topic switch. These routing probes must not change the
prompt interface: the prompt still receives exactly 10 candidates retrieved
from `common_terms + active_domain_overlay`.

Switch guards run in this order:

- no switch during warmup;
- update at most once per configured interval;
- no switch if the target is already the active preset or domain;
- no narrow switch to the general/common domain;
- no switch until the target index is preloadable;
- no switch unless confidence is above the threshold;
- no switch unless the new domain beats the next best domain by a margin;
- no switch unless the new domain also beats the current active slice by the
  current-margin threshold;
- no switch during the post-switch cooldown, so a just-applied slice cannot
  immediately ping-pong on the next retrieval tick;
- no switch unless the target is consistent across consecutive candidate
  windows; stale candidate streaks are reset;
- uncertain sessions stay on the current domain slice or no active fallback.

Manual glossary terms may still be injected into the prompt by
`PromptBuilder`, but they intentionally do not bias the router.

The default routing thresholds live in `configs/autoterm_slices.yaml`:

```yaml
routing:
  mode: hybrid_window_topic
  text_topic_weight: 0.60
  domain_probe_weight: 0.25
  domain_probe_top_k: 5
  speech_centroid_weight: 0.10
  metadata_prior_weight: 0.05
  domain_activate_threshold: 0.60
  domain_margin_threshold: 0.15
  current_margin_threshold: 0.10
  min_consistent_windows: 2
  min_consistent_windows_with_text: 2
  min_consistent_windows_audio_only: 3
  switch_cooldown_sec: 90
  candidate_stale_sec: 120
```

Each `topic_router` metadata payload records the target score, current active
slice score, target-current delta, candidate streak, top scored domains, and
the guard that blocked or allowed a switch. These fields are the first place to
inspect when a run appears to stay on a stale domain slice or switch too often.
JSON events also include `domain_probe_scores`, `domain_probe_slices`,
`domain_probe_cached`, and `domain_probe_s` when routing-only domain probes run
or cached probe evidence is reused for that chunk.

## Non-Blocking Switching

Cold index loads can take seconds for large memories. Adaptive switching avoids
blocking `_process_batch()`:

1. Retrieval returns references plus the pooled speech query embedding.
2. The router observes those signals and may emit a `switch` or `fallback`
   decision.
3. A per-session background task preloads the target index via
   `RetrievalPlugin.preload_index()`.
4. The session switches `active_glossary_preset` and `glossary_index_path` only
   after `is_index_ready()` returns true.
5. If a chunk arrives while the active auto index is still cold, retrieval for
   that chunk is skipped and the current translation path continues.

Manual preset activation can still warm the selected index through
`/glossary/build`, which is outside the streaming batch path.

## Prompt And UI Budgets

The retriever can return up to the UI budget, and prompt injection uses the same
default budget:

```text
RASST_PROMPT_TOP_K=10
RASST_UI_TOP_K=10
```

The JSON WebSocket event contains:

```json
{
  "type": "partial",
  "text": "...",
  "meta": {
    "references": [{"term": "...", "translation": "...", "source": "auto:nlp_core_10k"}],
    "prompt_reference_count": 10,
    "ui_reference_count": 10,
    "domain_probe_scores": {
      "medicine": {
        "preset_id": "medicine_core_10k",
        "top_score": 0.84,
        "mean_topk_score": 0.71,
        "top_terms": ["clinical trial", "patient"]
      }
    },
    "topic": {
      "active_domain": "medicine",
      "confidence": 0.73,
      "active_glossary_preset": "medicine_core_10k",
      "switch_count": 1
    },
    "topic_router": {
      "action": "switch",
      "from_preset": "nlp_core_10k",
      "to_preset": "medicine_core_10k",
      "confidence": 0.73,
      "margin": 0.19,
      "reason": "hybrid_window_topic",
      "evidence": {
        "router_text_source": "streaming_asr",
        "current_score": 0.41,
        "target_score_delta": 0.32,
        "candidate_preset": "medicine_core_10k",
        "candidate_streak": 2
      }
    }
  }
}
```

Plain-text WebSocket mode is unchanged.

## Runtime Configuration

`configs/autoterm_slices.yaml` is the source of truth for the default automatic
router. The current production defaults are:

```yaml
auto_working:
  prompt_k: 10
  base_slice: common_terms
  initial_slice: nlp_core_10k
  routing:
    mode: hybrid_window_topic
    text_topic_weight: 0.60
    domain_probe_weight: 0.25
    speech_centroid_weight: 0.10
    metadata_prior_weight: 0.05
    domain_activate_threshold: 0.60
    domain_margin_threshold: 0.15
    current_margin_threshold: 0.10
    min_consistent_windows_with_text: 2
    min_consistent_windows_audio_only: 3
    production_update_sec: 30
    production_warmup_sec: 20
    production_cooldown_sec: 90
```

`embedding_refs` and `legacy_keywords` remain explicit compatibility router
modes for debugging, but they are not the default auto-term strategy.

## Building Working Slices

CPU-side slice/manifest build:

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo
bash scripts/term_memory/build_domain_slices.sh \
  /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/source/wiki_translated.json \
  working_20260619
```

This writes:

```text
/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/glossaries/<slice>.zh.json
/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/snapshots/<snapshot>/<slice>.en-zh.jsonl
/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/manifests/current.json
```

Then build one MaxSim index per slice with the RASST text-index builder and put
it at:

```text
/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/<slice>/en-zh/maxsim.pt
```

Then build and publish router centroids:

```bash
python scripts/term_memory/build_domain_centroids.py \
  --manifest /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/manifests/current.json \
  --out-dir /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/centroids \
  --presets nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k \
  --target-lang zh \
  --update-manifest
```

Large artifacts stay under `/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime`,
not in this repo.

## Evaluation

Router-unit switch diagnostics:

```bash
python eval/streaming_sst/eval_auto_glossary_switch.py \
  --acl-text /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/source_text.txt \
  --medicine-text /mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh/medicine.source_text.en__medicine_404.txt \
  --max-windows-per-domain 8 \
  --max-switch-windows 4 \
  --out-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_switch_router_only_20260707.json
```

The 2026-07-07 Taurus source-text run passed with ACL->medicine switch latency
4 windows and medicine->ACL switch latency 2 windows. The medicine sample starts
with generic webinar operator lines before the oncology topic appears, so the
clean fixture regression remains stricter at 2 windows while the real-text run
uses a 4-window threshold.

This script directly drives `HybridWindowTopicRouter` on fixture/source-text
windows with wall-clock update and switch cooldown set to zero. It is a
router-unit diagnostic for the window-topic-first state machine, not an
end-to-end proof of live ASR, Omni batch-loop timing, real MaxSim domain-probe
quality, or production switch latency.

End-to-end streaming metrics:

```bash
python eval/streaming_sst/eval_auto_glossary.py \
  --base-url http://127.0.0.1:8011 \
  --seg-dir /mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_smoke/seg \
  --presets none,acl_tagged_raw,medicine_core_10k,nlp_core_10k,auto_working,open_wiki_100k \
  --out-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary.json
```

Optional term recall:

```bash
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
```

Combine as a table:

```bash
python eval/streaming_sst/score_auto_glossary.py \
  --auto-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary.json \
  --term-json /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_terms.json \
  --out-md /mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_table.md
```

## Failure Behavior

- Missing manifest: `auto_working` starts and falls back to `none` or mock
  indexes in `RASST_DEMO_MOCK=1`.
- Missing target slice: topic update records `target_unavailable` and keeps the
  current active glossary.
- Cold target index: preload happens in the background; the session switches
  only after the index is ready.
- Router exception: logged and ignored; translation continues.
- Retrieval failure at startup: existing graceful degradation path keeps the
  agent running without RAG.

## Paper Framing

The runtime budget is the fixed top-10 prompt list, not the full domain
universe. 100k/500k/1M memories remain useful as offline memory and scale
evidence, but the demo should select and rank candidates from an active
inventory. The claim is:

1. large open terminology memory can be maintained offline;
2. active inventory slices can be selected automatically from audio-native
   retrieval signals;
3. fixed top-10 reranking reduces distractors relative to direct broad-memory
   prompting;
4. users get terminology-aware streaming speech translation with zero setup.

Use this paper wording:

```text
RASST-Demo uses a lightweight, audio-native, confidence-gated active inventory
router. The router uses speech-side retrieval embeddings and retrieved-term
metadata to route directly among domain-specific slices only when the domain
evidence is strong. When evidence is ambiguous, it keeps the current domain slice
or no fallback active and preserves the fixed 10-candidate prompt interface.
```

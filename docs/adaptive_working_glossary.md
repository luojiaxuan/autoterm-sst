# Zero-Setup Adaptive Working Glossary

RASST-Demo maintains a large open terminology memory offline, but activates a
fixed 10-candidate prompt list for each streaming session. The active inventory is
selected automatically from speech-side retrieval evidence, so a user can start
a domain talk without uploading terms or choosing a domain preset.

## Architecture

```text
offline open memory
  -> working slices: common_10k / nlp_core_10k / medicine_core_10k / finance_core_10k / legal_core_10k
  -> manifest: preset id -> terms.jsonl + maxsim index + domain metadata + centroid
  -> runtime session starts in auto_working
  -> active preset starts as common_10k
  -> MaxSim retrieval exposes speech query embedding + retrieved refs
  -> AudioNativeActiveGlossaryRouter scores embedding-to-centroid + ref metadata
  -> ActiveGlossaryManager preloads and atomically activates target index
  -> future chunks retrieve from the active slice
  -> prompt receives top 10 refs, UI receives top 10 refs + router metadata
```

The framework boundary stays unchanged. `framework/app.py` and
`framework/router.py` still only move REST/WebSocket messages and
`TranslationEvent`s. All adaptive behavior lives inside `OmniAgent`, because
retrieval, prompting, batching, model state, and glossary selection are agent
concerns.

## Runtime Modules

- `framework/agents/term_memory/domain_taxonomy.py`: fallback/default domain
  labels and offline slice-ranking helpers. Runtime routing does not depend on
  keyword lists by default.
- `framework/agents/term_memory/topic_router.py`: audio-native active glossary
  router using speech query embeddings and retrieved-reference metadata. The
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
active_glossary_preset      # concrete retrieval preset, e.g. common_10k
active_domain               # general / nlp / medicine / finance / legal
topic_confidence
last_topic_update_s
topic_history
glossary_switch_count
recent_references
router_state                 # EMA speech embedding, recent decisions, pending target
last_router_decision
```

When the user selects a concrete preset (`none`, `acl_tagged_raw`,
`nlp_core_10k`, etc.), the session becomes manual and topic routing is disabled.
When the user selects `auto_working`, topic routing is enabled and the initial
active preset is `RASST_AUTO_GLOSSARY_DEFAULT` (`common_10k` by default).

## Routing Logic

The default router is `RASST_ROUTER_MODE=embedding_refs`. It is not a trained
topic classifier and it does not use ASR, source transcripts, generated target
text, LLM classification, or manual glossary terms for routing.

The router combines two signals:

```text
score(domain slice) =
    0.65 * cosine(EMA(speech_query_embedding), domain_centroid)
  + 0.35 * score-weighted retrieved-reference metadata votes
  + small consistency bonus
  - ambiguity penalty
```

Domain centroids are built offline from each preset's MaxSim text index:

```text
centroid = normalize(mean(normalize(text_embs), dim=0))
```

Reference votes use metadata such as `active_glossary_preset`, `domain`, and
`source_preset`; they do not infer topic from term strings like "patient" or
"language model".

Switch guards:

- no switch during warmup;
- update at most once per configured interval;
- no switch unless confidence is above the threshold;
- no switch unless the new domain beats the next best domain by a margin;
- no switch unless the target is consistent across recent decision windows;
- no narrow switch to the general/common domain;
- no switch until the target index is preloadable;
- uncertain sessions stay on `common_10k` or the current active preset.

Manual glossary terms may still be injected into the prompt by
`PromptBuilder`, but they intentionally do not bias the router.

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
    "topic": {
      "active_domain": "nlp",
      "confidence": 0.73,
      "active_glossary_preset": "nlp_core_10k",
      "switch_count": 1
    },
    "topic_router": {
      "action": "switch",
      "from_preset": "common_10k",
      "to_preset": "nlp_core_10k",
      "confidence": 0.73,
      "margin": 0.19,
      "reason": "speech_embedding+retrieved_refs"
    }
  }
}
```

Plain-text WebSocket mode is unchanged.

## Environment Knobs

```bash
export RASST_AUTO_GLOSSARY_ENABLED=1
export RASST_AUTO_GLOSSARY_DEFAULT=common_10k
export RASST_AUTO_GLOSSARY_PRESETS=common_10k,nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k
export RASST_AUTO_GLOSSARY_UPDATE_SEC=45
export RASST_AUTO_GLOSSARY_WARMUP_SEC=30
export RASST_AUTO_GLOSSARY_MIN_CONF=0.60
export RASST_AUTO_GLOSSARY_MIN_MARGIN=0.15
export RASST_AUTO_GLOSSARY_MIN_CONSISTENT_WINDOWS=2
export RASST_AUTO_GLOSSARY_FALLBACK=common_10k
export RASST_AUTO_GLOSSARY_PRELOAD=1
export RASST_AUTO_GLOSSARY_PRELOAD_PRESETS=common_10k,nlp_core_10k,medicine_core_10k
export RASST_ROUTER_MODE=embedding_refs
export RASST_ROUTER_LEGACY_KEYWORDS=0
export RASST_ROUTER_EMBED_WEIGHT=0.65
export RASST_ROUTER_REF_WEIGHT=0.35
export RASST_ROUTER_EMA_ALPHA=0.80
export RASST_PROMPT_TOP_K=10
export RASST_UI_TOP_K=10
export RASST_TERM_MEMORY_MANIFEST=/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/manifests/current.json
```

`auto_working` is the default when `RASST_AUTO_GLOSSARY_ENABLED=1`, unless
`RASST_DEFAULT_GLOSSARY_PRESET` explicitly overrides it.

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
  --presets common_10k,nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k \
  --target-lang zh \
  --update-manifest
```

Large artifacts stay under `/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime`,
not in this repo.

## Evaluation

Routing/latency/switch metrics:

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
metadata to add routed domain slices on top of common terms only when the domain
evidence is strong. When evidence is ambiguous, it keeps common terms active and
preserves the fixed 10-candidate prompt interface.
```

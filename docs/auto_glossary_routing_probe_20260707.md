# Auto Glossary Routing Probe 2026-07-07

本文记录 `auto_working` 从 ACL/NLP talk 切到 medicine talk 时的真实路由检查。结论先写在前面：旧版默认策略不是 window text/topic router，而是
`speech_query_embedding + retrieved-ref metadata` router；在真实 ACL -> medicine probe 中，它没有成功切到 `medicine_core_10k`。当前实现已经改成
`HybridWindowTopicRouter`：生产路径使用 speech-window domain probe 和 generated target translation window；source transcript window 只作为 controlled eval/diagnostic。

## 旧版实现问题

旧版默认 router 是 `framework/agents/term_memory/topic_router.py` 里的
`AudioNativeActiveGlossaryRouter`：

- 输入：MaxSim retriever 暴露的 speech-side query embedding，以及当前 active slice 检索出的 refs metadata。
- 默认分数：

```text
score(domain slice) =
    0.65 * cosine(EMA(speech_query_embedding), domain_centroid)
  + 0.35 * score-weighted retrieved-reference metadata votes
  + consistency bonus
  - ambiguity penalty
```

- 不是 ASR/source transcript topic classifier。
- probe 当时的 production code 只激活一个 domain-specific slice；当前策略继续保持
  domain-specific active inventory，不再默认 prepend common glossary。
- switch guard 包括 warmup、45s update interval、min confidence、min margin、current-margin、cooldown、连续候选 window。

这意味着如果当前 active slice 是 `nlp_core_10k`，retrieved-ref metadata 会自然偏向 NLP；跨域切换主要依赖 speech embedding 与各 domain centroid 的相似度。

## Probe 设置

本次不打断正在运行的完整 benchmark，只在 Taurus GPU4 上跑轻量 retriever probe。

本地临时输出：

| Artifact | Path | Status |
| --- | --- | --- |
| ACL->medicine router decision probe | `/mnt/data1/jiaxuanluo/rasst_eval/router_probe_acl_to_medicine_20260707_v2.json` | local staging |
| Per-window raw cosine probe | `/mnt/data1/jiaxuanluo/rasst_eval/router_probe_window_cosines_20260707.json` | local staging |
| Domain-probe retrieval-score probe | `/mnt/data1/jiaxuanluo/rasst_eval/router_probe_domain_retrieval_scores_20260707.json` | local staging |
| Segment source-text keyword probe | `/mnt/data1/jiaxuanluo/rasst_eval/router_probe_window_text_keywords_20260707.json` | local staging |

Data/model/index:

- ACL audio: `/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/audio/acl6060/2022.acl-long.367.wav`
- Medicine audio: `/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/audio/medicine/sample_404_v2/404_v2.wav`
- Medicine segment transcript: `/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh/medicine.source_text.en__medicine_404.txt`
- Retriever checkpoint: `/home/jiaxuanluo/rasst-demo/checkpoints/retriever/rasst-hn1024.pt`
- NLP index: `/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/wiki_nlp_ai_cs/en-zh/maxsim.pt`
- Medicine index: `/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/wiki_medicine/en-zh/maxsim.pt`

Each retrieval observation used a real streaming shape: 1.92s current chunk plus 1.92s lookback, not sentence-level batching.

## 结果

### 1. Current production router: no ACL -> medicine switch

Schedule: three ACL windows followed by four medicine windows in the same simulated session.

Result:

- Production decision actions: all `stay`.
- Production targets: all `nlp_core_10k`.
- Embedding-only variant: also all `nlp_core_10k`.

The medicine windows still had higher EMA cosine to the NLP centroid than to the medicine centroid after the ACL warmup windows.

### 2. Raw window cosine: centroid signal is weak

No EMA, direct per-window query embedding cosine to NLP and medicine centroids:

| Source | Windows | Predicted medicine | Avg `medicine_cos - nlp_cos` |
| --- | ---: | ---: | ---: |
| ACL | 13 | 5 | -0.0338 |
| Medicine | 17 | 2 | -0.0223 |

This is not usable as the main ACL-vs-medicine topic switch signal. Medicine windows were usually still closer to the NLP centroid.

### 3. Domain-probe retrieval score: useful but still noisy

Same audio window, retrieve against both NLP and medicine indexes and compare top retrieval score:

| Source | Windows | Predicted medicine | Avg medicine top-score delta |
| --- | ---: | ---: | ---: |
| ACL | 13 | 3 | -0.0461 |
| Medicine | 17 | 8 | 0.0112 |

This has more medicine signal than centroid cosine, but it is too noisy by itself. It can be a secondary signal after smoothing and margin checks.

### 4. Window source-text topic: strong signal

Using segment-level RASST source text around each audio window and the current domain keyword taxonomy:

| Source | Windows | Predicted NLP | Predicted medicine | Predicted general |
| --- | ---: | ---: | ---: | ---: |
| ACL | 13 | 7 | 0 | 6 |
| Medicine | 17 | 0 | 17 | 0 |

This validates source text as a diagnostic upper bound, but it is not the
deployable E2E signal because the demo does not run a separate ASR path.

## 下一版策略

`auto_working_v2` should be domain-specific and E2E-window-topic-first:

```text
speech-window domain probe
  + delayed generated target translation window
  -> text topic score over domain taxonomy and domain glossary aliases
  -> optional per-domain probe retrieval score
  -> speech embedding centroid score as weak tie-breaker
  -> hysteresis / consecutive-window switch guard
  -> switch exactly one active domain slice
  -> retrieve/rank prompt candidates from the active domain slice
  -> surface exactly 10 prompt candidates
```

Recommended score:

```text
score(domain) =
    0.60 * text_topic_score
  + 0.25 * domain_probe_retrieval_score
  + 0.10 * speech_centroid_score
  + 0.05 * metadata_prior
```

Before generated target text is available, route should fall back to domain-probe retrieval plus weak speech centroid, not rely on centroid alone.
The fallback is intentionally conservative: audio-only probe switches require
at least two positive probe-domain scores, a raw top score of at least `0.50`, a
raw top-vs-second margin of at least `0.08`, and agreement between the top probe
domain and the proposed target. Without generated target/source diagnostic text and without domain-probe
evidence, centroid similarity alone is not allowed to switch the active domain.

Recommended switch guard:

- Keep `min_consistent_windows = 2` for trusted source-text diagnostics.
- Use `min_consistent_windows_generated_target = 2` for generated target text
  in the current E2E setting. When the generated target window contains positive
  topic evidence, it is allowed to drive the switch even if the routing-only
  speech-window probe is weak or noisy. Probe evidence is still logged and used
  as auxiliary context; generic generated-target text without positive topic
  evidence still falls back to the stricter probe guard.
- Use `min_consistent_windows = 3` when only audio/probe signals are available.
- Keep cooldown to prevent ping-pong.
- Do not let current active-slice metadata vote veto a high-confidence text-topic switch; treat it as a small prior only.
- Probe retrieval must not change the prompt budget. It only scores candidate domains; prompt still receives exactly 10 candidates from the selected active slice.

## Open implementation work

- Generated target-translation text is now written into `router_text` after translation ticks as `router_text_source=generated_target`. Source transcript remains a controlled eval input, not the production E2E route.
- Routing-only domain-probe retrieval is now wired into Omni routing ticks for ready domain indexes. Fresh probe runs are gated by the router warmup/update/cooldown schedule, cached probe scores are reused during gate windows for router consistency, audio-only sessions refresh probes on the streaming window cadence instead of the full update interval, fresh probes reuse the main retrieval per-window speech embeddings when available, raw top-k probe scores are used rather than the prompt retrieval threshold, and `domain_probe_scores`, `domain_probe_slices`, `domain_probe_cached`, and `domain_probe_s` are recorded in JSON metadata without changing active prompt top-k.
- `eval/streaming_sst/eval_auto_glossary_switch.py` now provides router-unit ACL/NLP <-> medicine switch diagnostics using source-text windows or built-in fixtures. The latest Taurus source-text output is staged at `/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_switch_router_only_20260707_final8.json`; it passes all four scenarios with zero false switches in ACL-only and medicine-only streams. The clean fixture + contested-probe regression passes the stricter 2-window text-path threshold at `/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/eval/auto_glossary_switch_fixture_probe_20260707_final8.json`. This is not an end-to-end Omni/MaxSim deployment-latency benchmark, and speech/probe-only switching remains a guarded fallback before generated target evidence arrives.
- Continue expanding end-to-end active-slice candidate-quality eval with `router_text_source=generated_target` and speech-window domain probes; current source-text artifacts are diagnostic upper bounds, not generated-target E2E benchmark evidence.

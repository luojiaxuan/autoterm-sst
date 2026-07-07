# Auto Glossary Routing Probe 2026-07-07

本文记录 `auto_working` 从 ACL/NLP talk 切到 medicine talk 时的真实路由检查。结论先写在前面：当前默认策略不是 window text/topic router，而是
`speech_query_embedding + retrieved-ref metadata` router；在真实 ACL -> medicine probe 中，它没有成功切到 `medicine_core_10k`。

## 当前实现

当前默认 router 是 `framework/agents/term_memory/topic_router.py` 里的
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
- `auto_working` 当前只激活一个 domain-specific slice；不会 prepend common glossary。
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

- ACL audio: `/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/acl6060/2022.acl-long.367.wav`
- Medicine audio: `/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine/sample_404_v2/404_v2.wav`
- Medicine segment transcript: `/mnt/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh/medicine.source_text.en__medicine_404.txt`
- Retriever checkpoint: `/home/jiaxuanluo/rasst-demo/checkpoints/retriever/rasst-hn1024.pt`
- NLP index: `/mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/wiki_nlp_ai_cs/en-zh/maxsim.pt`
- Medicine index: `/mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/indexes/wiki_medicine/en-zh/maxsim.pt`

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

This matches the user's intuition: routing should primarily follow the topic expressed inside recent windows when source/ASR text is available.

## 下一版策略

`auto_working_v2` should be domain-specific and window-topic-first:

```text
recent source/ASR/topic text window
  -> text topic score over domain taxonomy and domain glossary aliases
  -> optional per-domain probe retrieval score
  -> speech embedding centroid score as weak tie-breaker
  -> hysteresis / consecutive-window switch guard
  -> activate exactly one domain-specific glossary slice
  -> retrieve/rank prompt candidates from active slice
  -> surface exactly 10 prompt candidates
```

Recommended score:

```text
score(domain) =
    0.55 * text_topic_score
  + 0.25 * domain_probe_retrieval_score
  + 0.15 * speech_centroid_score
  + 0.05 * metadata_prior
```

When no source/ASR/topic text is available, route should fall back to domain-probe retrieval plus weak speech centroid, not rely on centroid alone.

Recommended switch guard:

- Keep `min_consistent_windows = 2` for clear text-topic matches.
- Use `min_consistent_windows = 3` when only audio/probe signals are available.
- Keep cooldown to prevent ping-pong.
- Do not let current active-slice metadata vote veto a high-confidence text-topic switch; treat it as a small prior only.
- Probe retrieval must not change the prompt budget. It only scores candidate domains; prompt still receives exactly 10 candidates from the selected active slice.

## Open implementation work

- Add a runtime source for recent source/ASR/topic text. The current live pipeline does not expose English ASR/source transcript to the router.
- Add a domain-probe retrieval path that queries small top-k from candidate domain indexes for routing only.
- Reweight `AudioNativeActiveGlossaryRouter` or introduce `HybridWindowTopicRouter` with explicit signal diagnostics.
- Add ACL->medicine routing eval using RASST medicine HF/local data and assert switch latency, false-switch rate, and active-slice candidate quality.

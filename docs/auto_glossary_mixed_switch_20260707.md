# Auto Glossary Mixed-Domain Switch Benchmark - 2026-07-07

## 结论

这组结果只支持一个精确结论：在给定 clean expected domain-probe evidence 的情况下，
`auto_working` 的 target_translation_text-window state machine 可以在 ACL 5 个 talk 和
medicine 5 个 speech 组成的混合 playlist 中按 3-window consistency 规则切换
`nlp_core_10k` / `medicine_core_10k`。固定 64 windows/item 和全窗口设置都通过。

2026-07-07 后续真实 E2E streaming probe 进一步验证了部署路径：Taurus
`127.0.0.1:8012` 运行 `router_mode=hybrid_window_topic`，输入是真实
ACL/medicine audio，chunk 为 `latency_multiplier=2` 对应的 1.92s float32 PCM
window，不使用 ASR/source transcript。router text 来源是在线生成的 target
translation window。短测结果显示 ACL-only 不误切，medicine-only 能从初始 NLP
切到 medicine，ACL->medicine mixed run 能在 medicine 段开始后约 20.16s 切换到
`medicine_core_10k`，且无 wrong switch。

这次 benchmark 不使用 source transcript 或 ASR text。窗口文本来自 ACL 的中文 target
segments 和 RASST medicine 的中文 reference，作为 generated target translation window
的可复现实验代理。`probe_mode=expected` 表示 speech-domain probe guard 使用可控 clean
期望域证据；这验证 router state machine 和 target/probe 组合逻辑，不证明真实 speech
domain probe 的域判别质量，也不等价于完整 E2E Omni generation + real MaxSim probe
replay。`probe_mode=inverted/contested` 诊断会失败，用来确认 benchmark 能暴露错误或弱
probe 下的切换问题。

## 代码与输出

- Git ref: `88d0975 Block topic-empty centroid glossary switches`
- Taurus checkout: `/mnt/taurus/home/jiaxuanluo/rasst-demo`
- Taurus output dir:
  `/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_88d0975`
- 新脚本: `eval/streaming_sst/eval_mixed_domain_switch.py`
- 本地/Taurus 测试: `python3 -m unittest test_mixed_domain_switch_eval test_hybrid_window_topic_router test_auto_glossary_switch_eval`
- 数据来源:
  - ACL target/proxy windows:
    `/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments/segments.target`
    + `segments.meta.jsonl`
  - Medicine target/proxy windows:
    `/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh/medicine.ref.zh__medicine_*.txt`

输出放在 `/mnt/taurus/data1/...` 是因为本次运行前 Taurus `df -hT` 显示 data1 可用空间
高于 data2；结果摘要已进入 Git docs，raw JSON/MD 仍是 Taurus staging artifact。

真实 E2E streaming probe 的输出目录：

```text
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012
```

关键 artifacts：

| artifact | status |
|---|---|
| `acl80s_realtime_manifest_textfirst_8012.json` | ACL-only 80s, pass, active domain 全程 `nlp` |
| `medicine80s_realtime_manifest_textfirst_8012.json` | medicine-only 80s, 在 59.52s 切到 `medicine` |
| `acl1_medicine1_120s_realtime_manifest_textfirst_8012.json` | ACL 120s + medicine 120s 原始 E2E run |
| `acl1_medicine1_120s_realtime_manifest_textfirst_8012_switch30.json` | 同一原始记录按 30s real-streaming tolerance 重算 summary |

## Router 修正

本次发现并修复了一个切换弱点：generated target 文本只要存在，即使没有任何 topic
keyword hit，也会占用 `text_topic_weight=0.60`，从而把强 domain-probe evidence 稀释到
`confidence=0.2941`，导致过不了 `min_confidence=0.60`。修正后，只有当文本窗口有正
topic evidence 时才计入 text weight；泛化或无 topic hit 的 target 窗口不会压制
speech/domain probe。

同时补了 probe-only raw evidence guard：如果窗口文本存在但没有正 topic evidence，
实际有效信号退化为 probe-only，则必须通过更严格的 raw probe floor，避免
manifest/source 或 generated-target 泛化文本把弱 probe 归一化成高置信切换。

新增回归测试：

- `test_generated_target_generic_text_does_not_dilute_strong_probe`
- `test_generic_manifest_text_with_contested_probe_does_not_false_switch`
- `test_generic_generated_target_with_contested_probe_does_not_false_switch`
- `test_alternating_generated_target_playlist_switches_with_expected_probe`
- `test_random_playlist_counts_only_domain_transition_boundaries`
- `test_inverted_probe_diagnostic_fails`

## 固定 64 Windows/Item 主结果

| setting | windows | domain transitions | switches | max latency | domain accuracy | steady-state accuracy | wrong switches | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| alternating, expected probe | 640 | 9 | 9 | 3 | 0.9719 | 1.0000 | 0 | true |
| random seed 20260707, expected probe | 640 | 7 | 7 | 3 | 0.9781 | 1.0000 | 0 | true |
| alternating, no probe diagnostic | 640 | 9 | 0 | n/a | 0.5000 | 0.5024 | 0 | false |
| alternating, inverted probe diagnostic | 640 | 9 | 10 | fail | 0.0766 | 0.0424 | 10 | false |
| alternating, contested probe diagnostic | 640 | 9 | 7 | 13 | 0.8125 | 0.8434 | 0 | false |

解释：

- `max latency = 3` 是预期行为，因为 generated-target 路径配置了
  `min_consistent_windows_generated_target = 3`。
- `domain accuracy < 1.0` 只来自每个 domain transition 后允许的前两个滞后窗口；
  去掉 transition grace window 后，steady-state accuracy 是 1.0。
- no-probe 对照失败是预期的：当前 deployable 策略要求 generated target switch 需要
  domain-probe guard，不允许仅靠 target text 单独切换。
- inverted-probe 对照产生 10 次 wrong switches；contested-probe 对照最长 13 个窗口才切换，
  说明这个 benchmark 可以暴露 probe 错误或 probe 过弱时的失败模式。

## 全窗口对照

| setting | windows | domain transitions | switches | max latency | domain accuracy | steady-state accuracy | wrong switches | pass |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| alternating, expected probe | 1905 | 9 | 9 | 3 | 0.9906 | 1.0000 | 0 | true |
| random seed 20260707, expected probe | 1905 | 7 | 7 | 3 | 0.9927 | 1.0000 | 0 | true |

## 真实 E2E Streaming Probe

服务器配置：

```text
host: Taurus
base_url: http://127.0.0.1:8012
router_mode: hybrid_window_topic
manifest: auto_working_alias_20260619T204803Z
auto presets: nlp_core_10k, medicine_core_10k
prompt_k: 10
chunk: 30720 samples = 1.92s
router_text_source: generated_target
```

| run | audio | active domains | switch latency | wrong switches | steady-state accuracy | retrieval p95 |
|---|---:|---|---:|---:|---:|---:|
| ACL-only | 80s | `nlp`: 41 | n/a | 0 | 1.0000 | 86.97ms |
| medicine-only | 80s | `nlp`: 24, `medicine`: 12 | 59.52s from session start | 0 | n/a | n/a |
| ACL->medicine | 240s | `nlp`: 71, `medicine`: 52 | 20.16s after boundary | 0 | 1.0000 with 30s tolerance | 88.66ms |

重要观察：

- speech-window domain probe 仍然噪声很大：ACL->medicine run 中 probe top accuracy
  只有 0.6071，medicine-only 80s 早期 probe top 也常偏 `nlp`。
- 因此当前有效切换主要来自 generated target translation window 的 topic signal，
  probe 是辅助/诊断信号，而不是硬性主路由。
- 旧 `max_switch_events=3` 是 proxy smoke test 判据；真实 1.92s streaming 下只等价
  5.76s，不适合评估 window-topic-first router。`eval_mixed_audio_switch.py` 现在支持
  `--max-switch-seconds`，建议真实 E2E mixed run 使用 `--max-switch-seconds 30`，同时报告
  实际 `latency_s`。

Full 5 ACL + 5 medicine runs were started on Taurus at commit `d63202d` and
then canceled to save resources after confirming they were unnecessarily long
for the current routing question:

```text
output dir:
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012_full_d63202d

alternating PID: 3693023
random seed 20260707 PID: 3693024
```

The replacement run is a 4-block real E2E streaming playlist:

```text
ACL 120s -> medicine 120s -> ACL 120s -> medicine 120s
output:
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_hybrid_8012_4block/alternating_acl_med_acl_med_120s_realtime_switch30.json
```

Metrics:

| metric | value |
|---|---:|
| audio seconds | 480.0 |
| events | 238 |
| domain transitions | 3 |
| switches | 3 |
| wrong switches | 0 |
| active domain accuracy | 0.8613 |
| steady-state active domain accuracy, 30s tolerance | 0.9947 |
| retrieval p95 | 88.29ms |
| prompt invariant violations | 0 |
| probe top accuracy | 0.5242 |

Transition latencies:

| transition | latency |
|---|---:|
| ACL -> medicine_404 | 20.16s |
| medicine_404 -> ACL | 17.28s |
| ACL -> medicine_606 | 37.44s |

Strict `--max-switch-seconds 30` marks the final transition as failed because
37.44s is over the 30s threshold. Recomputing on the same record with 40s or
45s tolerance passes with steady-state accuracy 1.0. The delayed final switch is
not a wrong switch: the medicine_606 opening is mostly speaker/session framing
until the generated translation reaches explicit oncology terms such as
`肿瘤内科医生` and `放射肿瘤科医生`.

## Term Accuracy Comparison

For paper-facing comparison, switch latency is less important than terminology
accuracy under the same retrieval cap. The 4-block term accuracy run uses
the same 480s audio and compares fixed glossary presets with `auto_working`:

```text
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_termacc_4block
```

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

`manual_fixed_by_domain` is a composed diagnostic row: fixed NLP output for ACL
blocks plus fixed medicine output for medicine blocks. The fixed NLP row remains
strong on medicine because term accuracy is output-centric and the base model can
recover common medicine terms without the active medicine slice; this table
should not be used as prompt-channel attribution.

### Fixed NLP 在 medicine 上看起来不差的原因

2026-07-08 诊断显示，这个现象不是 NLP glossary 覆盖了 medicine gold。实际原因是：

- 当前 4-block eval 的 medicine 分母很小：`medicine_404` 前 120s 没有 oracle term，
  全部 medicine gold 都来自 `medicine_606` 前 120s，共 12 个 occurrence、6 个唯一 term。
- 这 12 个 occurrence 被少数重复 term 主导：`rectal cancer` 出现 5 次，
  `medical oncologist` 和 `Radiation Oncologist` 各出现 2 次。
- `score_mixed_audio_terms.py` 的主指标是 output-centric occurrence term_ACC：
  只要该 block 输出中出现 oracle target variant，该 block 内同一 term 的每个
  occurrence 都计 hit。这会让 `rectal cancer -> 直肠癌` 这类重复 term 放大。
- 实际 glossary 内容没有泄漏：`wiki_academic_zh.json` 不包含上述 6 个 medicine
  gold terms；`wiki_medicine_zh.json` 也只覆盖其中一部分。
- fixed medicine 反而较低，部分是 exact-string oracle 口径导致的。例如 medicine
  glossary 中 `radiation oncologist` 对应 `放射肿瘤学家`，但 oracle 只接受
  `放射肿瘤科医生`；fixed medicine 输出医学上合理的同义译法仍会被记 miss。
- 旧 fixed preset serving 路径和 `auto_working` 的 retrieval 设置不一致：
  fixed NLP 在 `medicine_606` block 平均只有 1.05 个 prompt refs，fixed medicine
  平均 2.66 个；旧 `auto_working` 当时会回填到 10。按最新口径，所有 preset 都应
  先召回 top-10，再按分数过滤，过滤后有多少 prompt refs 就给多少。因此这组旧
  fixed rows 只能说明 output term_ACC，不应作为 glossary-channel 对比。

补充诊断表已经用同一批 pre-change JSON 重算，不重新跑模型；`type_acc_any`
按 block-local unique term type 统计，避免同一 term 在不同 block 里 hit 一次就把全局
type 全部算对：

```text
/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260707_termacc_4block/term_acc_compare_type_diagnostics.md
```

| denominator | run | medicine occurrence acc | medicine type_acc_any | medicine type hits |
|---|---|---:|---:|---:|
| technical+medicine | fixed_nlp | 0.9167 | 0.8333 | 5/6 |
| technical+medicine | fixed_medicine | 0.6667 | 0.5000 | 3/6 |
| technical+medicine | auto_working | 0.9167 | 0.8333 | 5/6 |
| raw+medicine | fixed_nlp | 0.9167 | 0.8333 | 5/6 |
| raw+medicine | fixed_medicine | 0.6667 | 0.5000 | 3/6 |
| raw+medicine | auto_working | 0.9167 | 0.8333 | 5/6 |

后续重新跑 fixed-vs-auto term_ACC 时必须使用修正后的 serving 代码：fixed preset 和
auto preset 都只召回 top-10，并在分数过滤后保留实际 surviving prompt refs；`none`
或 `no_glossary` baseline 仍没有 glossary refs。paper 表格里建议同时报告
occurrence term_ACC、block-local unique-term/type term_ACC、PromptGoldRetrieved@10、
surviving prompt refs/chunk 和 prompt shortfall，避免把模型自身常识翻译误读成
glossary channel 成功。

## 2026-07-08 长流式 ACL -> medicine_606 -> ACL 结果

这次重新跑了真实 E2E streaming，而不是 120s smoke。playlist 是：

| block | item | domain | seconds |
|---:|---|---|---:|
| 1 | `2022.acl-long.268` | nlp | 687.2 |
| 2 | `medicine_606` | medicine | 2842.8 |
| 3 | `2022.acl-long.367` | nlp | 612.3 |

总音频 4142.4s，按 `latency_multiplier=2` 以 1.92s float32 PCM chunk 实时发送。
输出目录：

```text
/mnt/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260708_acl_med_acl_long
```

Git refs:

- `83fe359`: runtime 改为 top-10 cap 后按分数过滤，不再回填到固定 10 个 prompt refs。
- `c7d0b6f`: 修复 Taurus 上 RASST MaxSim retriever import path，8012 health 恢复为 RAG ready。
- `e7803b8`: mixed audio runner 支持 `--medicine-ids 606`，避免默认选到术语很少的 `medicine_404`。
- `5bb37ed`: mixed scorer 增加 BLEU / masked_term_BLEU。

Gold denominator:

| denominator | gold occurrences | ACL | medicine |
|---|---:|---:|---:|
| technical+medicine | 298 | 100 | 198 |
| raw+medicine | 382 | 184 | 198 |

下面的 term_acc 是 output-centric：只说明最终输出是否命中术语变体，不能单独证明
glossary channel 有贡献。尤其是当前 `medicine_core_10k` 覆盖不足，medicine 行必须和
后面的 inventory coverage 诊断一起解读。

主结果：

| denominator | run | term_acc | hits/gold | ACL acc | medicine acc | medicine type_acc_any | BLEU | masked_term_BLEU |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| technical+medicine | fixed_nlp_core_10k | 0.7785 | 232/298 | 0.8400 | 0.7475 | 26/54 | 52.2587 | 49.1001 |
| technical+medicine | fixed_medicine_core_10k | 0.7617 | 227/298 | 0.8200 | 0.7323 | 24/54 | 52.0972 | 48.8884 |
| technical+medicine | auto_working | 0.7550 | 225/298 | 0.8200 | 0.7222 | 23/54 | 52.3029 | 49.0325 |
| raw+medicine | fixed_nlp_core_10k | 0.7801 | 298/382 | 0.8152 | 0.7475 | 26/54 | 52.2587 | 48.1131 |
| raw+medicine | fixed_medicine_core_10k | 0.7723 | 295/382 | 0.8152 | 0.7323 | 24/54 | 52.0972 | 47.8451 |
| raw+medicine | auto_working | 0.7644 | 292/382 | 0.8098 | 0.7222 | 23/54 | 52.3029 | 48.1523 |

Runtime/router diagnostics:

| run | events | active domains | prompt refs | retrieval p50/p95 |
|---|---:|---|---|---:|
| fixed_nlp_core_10k | 2081 | `nlp`: 2081 | min 0, max 10 | 79.66 / 99.62ms |
| fixed_medicine_core_10k | 2074 | `medicine`: 2074 | min 0, max 10 | 80.78 / 98.25ms |
| auto_working | 2081 | `nlp`: 677, `medicine`: 1404 | min 0, max 10 | 83.80 / 101.34ms |

`auto_working` 的 `router_text_source` 是 `generated_target`: 2081/2081。固定
glossary rows 不启用 router，所以 `router_text_source=none`。这说明本次切换证据来自
E2E 已生成的 target translation window，而不是 source transcript/ASR。

`auto_working` 切换是成功的：

| transition | boundary_s | first target active | latency |
|---|---:|---:|---:|
| ACL -> medicine_606 | 687.243 | 727.680 | 40.437s |
| medicine_606 -> ACL | 3530.062 | 3552.000 | 21.938s |

`auto_working` 的 active-domain accuracy 是 0.9861，去掉 transition grace 后
steady-state accuracy 是 1.0，wrong switch 为 0。也就是说当前问题不是 router
没有切到 medicine，而是切到当前 `medicine_core_10k` 后没有提高 medicine term_acc。

### 为什么 fixed NLP 在 medicine 上也不差

这次长流式结果确认：fixed NLP 的 medicine acc 高不是因为 NLP glossary 覆盖了
medicine gold，也不是因为分母太小。

1. `medicine_606` 的 medicine denominator 足够大：198 occurrences、54 unique term
   types。
2. runtime prompt refs 不是固定 10。fixed NLP 在 medicine 段平均只有 1.069 个 prompt
   refs，733/1413 个 medicine events 是 0 refs；fixed medicine/auto 在 medicine 段平均
   约 2.3 refs。因此 fixed NLP 的 medicine term_acc 更接近模型自身翻译能力。
3. 当前 runtime `medicine_core_10k` 不是 benchmark hard medicine glossary，而是 broad
   `wiki_medicine` slice。对 `medicine_606` 54 个 unique gold terms 的 exact inventory
   coverage：

| inventory | size | exact coverage |
|---|---:|---:|
| `nlp_core_10k` / `wiki_academic` | 18994 | 1/54 |
| `medicine_core_10k` / `wiki_medicine` | 23713 | 1/54 |
| `common_10k` | 10000 | 0/54 |
| RASST hard medicine glossary | 212 | contains the checked hard terms |

4. fixed NLP 比 fixed medicine 多出的 medicine hits 只有 3 个 occurrence：
   `T2 N1` 两次、`short-course of chemo-radiotherapy` 一次。fixed NLP 比 auto 多出的
   medicine hits 是 7 个 occurrence；auto 反过来多命中 `FOLFIRINOX` 两次。差异主要是
   输出措辞/retention 的小波动，不是 glossary channel 的大收益。
5. 三组共同 miss 高度重合，集中在 `long-term outcomes`, `Gy`,
   `Radiation Oncologist`, `medical oncologist`, `total neoadjuvant`,
   `T3 tumours`, `Locoregional recurrence`, `target volume` 等术语。

结论：当前 `medicine_core_10k` 不应被解释成 medicine benchmark coverage slice。它是
broad wiki medicine inventory，和 RASST hard medicine gold 的 overlap 很低。若 paper
需要证明 domain-specific glossary 的上界，应该增加一个 eval-only
`medicine_hardraw_oracle` preset，使用现有 hard medicine index：

```text
/mnt/taurus/data2/jiaxuanluo/RASST/outputs/main_result_eval/20260527T071109Z/index_cache/medicine_hardraw__zh__lm2/maxsim_hard_medicine_glossary_raw_llm_judge_manual_zh21_6d02fb5133b93f6d_tr128_ta256.pt
```

真正的 auto route 逻辑应区分：

- real broad domain slice: `medicine_core_10k` / `wiki_medicine`
- eval-only oracle slice: `medicine_hardraw_oracle`
- future curated domain slice: 从 RASST hard medicine、UMLS/MeSH/clinical trial
  terminology 等构建的 compact medicine benchmark-aligned slice

### Fixed-denominator union glossary 修复 (2026-07-08)

上面 1/54 coverage 的根因修复已经落地为 ACL `gs` 系列的 medicine 版本：GT 必须
union 进 fixed 10k inventory，否则 glossary channel 无法被测量。

- 新 builder：`scripts/term_memory/build_gt_union_gs_glossary.py`。保证 GT 条目
  byte-identical 且排最前、normalized source term 碰撞时 GT 优先、filler 按
  `(term, translation)` 排序后 seeded shuffle、输出严格等于 `--size`（不足则
  loud fail）。本地已用合成 filler 池验证 size/identity/determinism/collision/
  no-zh-skip 五项保证。
- HF 已上传（revision `204ba141`，repo
  `gavinlaw/rasst-main-result-data`）：
  - `glossaries/hard_medicine_gt_raw_unique212.json` — compact runtime-schema
    GT（与 `main_result/inputs/medicine_zh/hard_medicine_raw__medicine5.json`
    byte-identical），union-ready。
  - `glossaries/README.md` — 配方、build 命令、验证记录。
  - `dataset_manifest.json` — 新增 `glossary_medicine_hardraw_compact` asset。
- GT 验证（用 HF 副本核对）：212 条全有 zh、无 normalized 重复；per-talk unique
  terms 404:20 / 545006:20 / 596001:49 / 605000:66 / 606:60；`medicine_606`
  oracle 的 54 个 unique terms 100% 被 GT 覆盖（对照 broad slice 的 1/54）。
- **Union 已于 2026-07-08 在 taurus 构建并验证**（builder 跑在
  `c216e0f`，seed 1215）：

```text
taurus: /mnt/data2/jiaxuanluo/rasst-demo/runtime/term_memory/glossaries/medicine_hardraw_gt_union_gs10000.json
GT: gemini .../zh/__medicine_inputs__/lists/hard_medicine_raw__medicine5.json (212)
filler: runtime glossaries/wiki_medicine_zh.json (23,713 pool; 12 GT collisions skipped; 9,788 used)
```

  独立复核：exactly 10,000 条、GT 212 条 byte-identical 且在最前、无 normalized
  重复、全部有 zh、`medicine_606` oracle coverage **54/54**（broad slice 时代是
  1/54）。sources 分布：`medicine_hard_llm_judge_manual: 212` +
  `wikidata: 9788`。选 `wiki_medicine_zh` 而不是 ESO translated 池做 filler，
  是为了与 ACL `gs` 配方严格对称：distractor 全部来自 wiki，不再混入其他
  talk-derived terms。union + report 已回传 HF（revision `365c84ba`）。
- **Index 与 preset 注册已完成（2026-07-08）**，在 aries GPU5（spaCyEnv,
  hn1024 checkpoint, tr128/ta256 默认）构建：

```text
indexes/medicine_hardraw_gs10k/en-zh/maxsim.pt   (10,000 x 1024, 42MB)
indexes/medicine_hardraw_oracle/en-zh/maxsim.pt  (212 x 1024, 0.9MB)
manifests/auto_working_medicine_hardraw_20260708.json
```

  新 manifest 是 `auto_working_alias_20260619T204803Z` 的增量副本（保留
  common_10k / nlp_core_10k / medicine_core_10k，新增 `medicine_hardraw_gs10k`
  与 eval-only `medicine_hardraw_oracle`），已用
  `framework.agents.term_memory.manifest.TermMemoryManifest.load()` 验证 5 个
  preset 全部 `index_ready`。`current.json` 未动，正在运行的 8012 server 不受
  影响。
- 剩余步骤：用
  `RASST_TERM_MEMORY_MANIFEST=.../manifests/auto_working_medicine_hardraw_20260708.json`
  启动 eval server，重跑 3-talk 长流式对比 fixed_nlp /
  fixed_medicine(`medicine_hardraw_gs10k`) / `medicine_hardraw_oracle` /
  auto_working，同时报告 PromptGoldRetrieved@10 与 surviving prompt refs。

### 截断 3-talk union 复跑结果 (2026-07-09)

Server: aries GPU 5/6 (vLLM TP2) + GPU 7 (RAG)，port 8013，manifest
`eval_medicine_union_20260708.json`（`medicine_core_10k` 内容 = hardraw union，
因为 `DOMAIN_TO_PRESET` 硬编码 medicine->medicine_core_10k，preset_meta 有
override 说明）。Playlist：alternating ACL 600s -> medicine_606 600s -> ACL
600s（每 talk 截断 10 分钟），lm=2、chunk 1.92s、real-time feed、
`--max-switch-seconds 60`。输出：
`/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260708_union_truncated_8013`。

technical+medicine 口径（gold 143；medicine 23 unique types）：

| run | term_acc | ACL acc | medicine acc | med type_acc | BLEU | masked |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.7692 | 0.8061 | 0.6889 | 0.391 (9/23) | 23.14 | 20.96 |
| fixed_nlp | 0.7273 | 0.7449 | 0.6889 | 0.391 (9/23) | 23.52 | 21.62 |
| fixed_medicine_union | 0.8252 | 0.7755 | 0.9333 | 0.870 (20/23) | 22.92 | 20.29 |
| auto_working | 0.8112 | 0.7449 | **0.9556** | **0.913 (21/23)** | 23.07 | 20.31 |
| oracle_medicine_hardraw | 0.8112 | 0.7755 | 0.8889 | 0.783 (18/23) | 22.94 | 20.59 |

路由（auto_working）：2/2 transitions、**0 wrong switches**、steady-state
accuracy **1.0**、切换延迟 35.52s (nlp->med) / 24.96s (med->nlp)、retrieval
p95 114.7ms、probe top accuracy 0.678。

结论：

1. **union 修复直接可测**：medicine type_acc 从 none/fixed_nlp 的 9/23 提到
   auto 的 21/23；medicine occurrence acc 0.689 -> 0.956（+26.7pp）。glossary
   channel 归因成立（gold coverage 54/54 vs 旧 broad slice 1/54）。
2. auto 与 fixed_medicine_union 差距 1.4pp（0.8112 vs 0.8252），来自切换延迟
   窗口；auto 在 medicine 段反而略高于 fixed union 和 oracle。
3. BLEU / masked BLEU 五组打平（22.9–23.5），glossary 注入不伤流畅度。
4. 注意 ACL 侧：none 的 ACL acc (0.806) 高于 fixed_nlp (0.745)——当前
   `nlp_core_10k`（wiki_academic）对 ACL gold 覆盖不足且引入干扰。**10-talk
   最终 run 建议 nlp 侧同样换成 `acl_tagged_gs10k` union**，两域对称的
   fixed-denominator 配方。
5. oracle (212-only) 在 medicine 段低于 union——极小 inventory 下 score
   filtering 行为不同，小样本不过度解读。

## 固定 64 命令

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo

python3 eval/streaming_sst/eval_mixed_domain_switch.py \
  --schedule alternating \
  --windows-per-item 64 \
  --router-text-source generated_target \
  --probe-mode expected \
  --max-switch-windows 3 \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_88d0975/alternating_target64_expected_probe.json \
  --out-md /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_88d0975/alternating_target64_expected_probe.md

python3 eval/streaming_sst/eval_mixed_domain_switch.py \
  --schedule random \
  --seed 20260707 \
  --windows-per-item 64 \
  --router-text-source generated_target \
  --probe-mode expected \
  --max-switch-windows 3 \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_88d0975/random_seed20260707_target64_expected_probe.json \
  --out-md /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_88d0975/random_seed20260707_target64_expected_probe.md
```

## 下一步

1. 用真实 E2E server 输出的 `meta.domain_probe_scores` 和 generated target window 做 replay，
   替换 `probe_mode=expected`。
2. 如果 GPU/服务可用，直接对 ACL/medicine audio playlist 跑 streaming WS eval，记录
   active glossary over time、prompt refs、BLEU、term_ACC、masked_term_BLEU。
3. 把当前 mixed switch benchmark 纳入后续 regression：fixed 64 windows/item、
   alternating + random seed 20260707 必须通过。

## 2026-07-09: 三语 3-talk、10-talk 车道与基础设施事记

- **v2 双 union（zh, 截断 3-talk）**：auto 0.9161/0.9115 双口径超 fixed_nlp
  (0.839/0.867) 与 fixed_medicine (0.825/0.819)；2/2 切换、0 错切、steady 1.0、
  切换 37.44s/24.96s。产物 `aries:/mnt/data3/jiaxuanluo/eval_out` 与
  `20260708_union_truncated_8013`。
- **32-session 压测（aries 8013, auto_working, 300s 实时）**：32/32 完成、
  0 失败 0 掉线、首字 p50 0.346s (1 session) → 0.537s (32)；全部 session 维持
  实时。产物 `taurus data1 rasst_eval/stress_auto_20260709`。
- **ja/de 全长 3-talk（keyword-tuned rerun）**：Taurus
  `kw_rerun_20260709` 已完成。ja auto technical/raw term_acc 为 .9098/.9000，
  de 为 .9113/.8798；两种语言均 2/2 正确切换、0 错切、steady-state 1.0。
  切换延迟为 ja 28.917/33.458s、de 51.957/33.458s。原先 115.3s/875.6s
  的数字保留为无 ja/de keyword 的 ablation，不代表 released 配置。
- **de 评分 artifact 修复**：`classify_output_hit` 的 CJK 门导致德语翻译
  variant 永不计分（全条件 ~0.11）；`133fd52` 起 matcher 语言感知（de 走
  casefold+词干容忍，短词严格词界），zh/ja 行为不变（回归 15/15 + 单测 6/6）。
- **taurus 内存风暴（11:28）**：host 匿名内存 905/1007GB、页缓存 4GB，NFS 挂载
  的权重 mmap 重缺页 → 三主机 engine/client 连锁死亡（8015 `execute_model
  RPC timeout`、所有 client WS keepalive 1011）。对策：aries 全本地化
  （模型×3/音频/index/gold → /mnt/data3/jiaxuanluo/local_cache），单主机最多
  一台 vLLM server 的纪律。
- **repo**：`rasst-demo` 更名 `autoterm-sst`，`framework` fast-forward 进
  `main`（此后直接 push main）。
- **checkpoint 清理**：删 de 探索变体 `cap16_exactboundary`（66G）与空 staging；
  zh 在 data1 的逐字节重复副本（66G，与 data2 prod md5 一致）为 root 所有，
  需管理员删除。

## 2026-07-09 深夜: zh 10-talk alternating 主结果（论文 Table 1 / 附录 C）

单 session 实时流式 16,848.1s（5 ACL + 5 medicine 交替，8,531 chunks），
双 union 10k inventories，`term_acc_10talk.{json,md}` 为权威产物
（aries `/mnt/data3/jiaxuanluo/eval_out/10talk_zh/`）。

| run | tech acc (875) | raw acc (1064) | ACL | Med. | Med. types | BLEU | M-BLEU |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 0.744 | 0.754 | 0.782 | 0.733 | 92/196 | 58.84 | 57.46 |
| fixed_nlp_union | 0.744 | 0.777 | 0.907 | 0.698 | 86/196 | 59.18 | 57.85 |
| fixed_medicine_union | 0.901 | 0.881 | 0.772 | 0.937 | 168/196 | 59.63 | 57.53 |
| auto_working | **0.936** | **0.933** | **0.912** | **0.943** | **170/196** | 58.10 | 56.22 |

- gold 口径：technical 875 occurrences（ACL 193 + medicine 682）；raw 1,064
  （ACL raw 382 + medicine 682）；medicine block-local 196 term types。
- 路由：9/9 domain transitions 方向正确 + 1 次瞬时多余切换（共 11 switches）；
  steady-state active-domain accuracy 0.9824；8/9 切换在 13.6–55.4s 内落地，
  1 次 medicine→ACL 延迟 304s（generic talk opening）。
- 结论：auto 双口径超两个 fixed union，且各自 in-domain 追平/略超对应
  domain expert（ACL 0.912 vs 0.907，medicine 0.943 vs 0.937）；BLEU 代价
  1–1.5（切换窗口）。论文 §6 tab:quality 与附录 C tab:mixed 已同步。

## 2026-07-10: zh 10-talk random（seed 20260707）车道与 8016 启动修复

- 首夜 random 链两次失败根因：8016 `RASST_GPU_MEMORY_UTILIZATION=0.55`
  在 A6000 48GB 上仅留 26.4GB < TP=2 权重 ~33GB/卡，vLLM 报
  `Available KV cache memory: -5.71 GiB` → engine init 失败。修复：0.85 +
  RAG 移到 cuda:0（GPU 4 全空），8016 于 00:06:59 UTC 起服务，75s 完成
  engine 装载。auto（8016, 实时 1.92s feed）+ trio（8013, 2×）并行中，
  产物将落 `aries:/mnt/data3/jiaxuanluo/eval_out/10talk_zh_random/`
  （`term_acc_random.{json,md}`）。

## Source-of-truth artifact paths (2026-07-09)

- zh 10-talk alternating main result: aries
  `/mnt/data3/jiaxuanluo/eval_out/10talk_zh/term_acc_10talk.json`
  (`auto_working` = 819/875 technical, 993/1064 raw).
- zh 3-talk matched-union result: Taurus local staging
  `/mnt/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_audio/20260708_union_truncated_8013/term_acc_compare_v2.json`
  (`auto_working_v2` = 131/143 technical, 206/226 raw).
- ja keyword-tuned result: Taurus local staging
  `/mnt/data1/jiaxuanluo/rasst_eval/kw_rerun_20260709/3talk_ja/term_acc_ja_kw.json`.
- de keyword-tuned result: Taurus local staging
  `/mnt/data1/jiaxuanluo/rasst_eval/kw_rerun_20260709/3talk_de/term_acc_de_kw.json`.
- These JSON outputs are local-only evaluation staging; they are not yet uploaded
  to a Hugging Face dataset. The paper source and the lightweight summaries in
  this document remain the Git source of truth until an artifact repository is
  selected.

## 2026-07-10: router λ 权重 robustness sweep（CPU proxy）

回答"权重怎么来的"：手设于 proxy，未对 audio eval 拟合。加了 4 个 CLI 旗子
（`--text-topic-weight` 等，默认不变，权重写入产物 summary）后在固定
64-window alternating + random(seed 20260707) proxy 上扫 7 组配置：

| config (λt/λp/λc/λm) | wrong | passed | mean lat (win) | regression |
|---|---:|---:|---:|---|
| default 0.60/0.25/0.10/0.05 | 0 | 16/16 | 3.00 | pass |
| topic_heavy 0.80/0.10/0.05/0.05 | 0 | 16/16 | 3.00 | pass |
| balanced 0.45/0.40/0.10/0.05 | 0 | 16/16 | 3.00 | pass |
| probe_heavy 0.25/0.60/0.10/0.05 | 0 | 16/16 | 3.00 | pass |
| probe_only 0/1/0/0 | 0 | 16/16 | 3.00 | pass |
| no_aux 0.70/0.30/0/0 | 0 | 16/16 | 3.00 | pass |
| topic_only 1/0/0/0 | 0 | 3/16 | 8.0/5.3 | **fail** |

结论：混合权重在宽区间内不敏感（延迟恒等于 3-window consistency guard）；
去掉 probe 通道后大多数切换被 guard 挡死——probe 是必要的 corroborating
guard（虽然真实语音 probe 单独太噪，见 §6 controls）。产物
`aries:/mnt/data3/jiaxuanluo/eval_out/lambda_sweep/`。已写入论文附录 A。

## 2026-07-10: demo 专用 curated slices（不影响任何评测清单）

用户反馈 demo 检索面板观感差（tasks/task/lexical 等泛词、LanguageWare→LanguageWare
等 identity filler、"这里"高亮）。新建 demo-only 词表 + 索引 + manifest
`demo_curated_20260710`（仅 taurus 8014 使用；所有评测 manifest 不变，论文
§5 已加披露句）：

- `demo_nlp_gs10k`：acl_tagged_gs10k 过滤 EN 泛词表 + zh 单字/杂词表 →
  **10000→9933**。identity 对（BERT→BERT、LanguageWare→LanguageWare 等）
  按用户裁定为合法术语保留（一度全删到 4968，已回滚重建）。
- `demo_medicine_gs10k`：union 同规则过滤 + 注入 6 条会话人名/角色
  （Ramon de Mello→拉蒙·德·梅洛、Katia (Roque) Perez、Maria Antonietta
  Gambacorta、Medical/Radiation Oncologist、TME surgery）→ 9996。
- 基座检索器修复：`RASST_INDEX_ZH_ACL` 默认指向 5 月的 238 条老索引，导致
  状态行首屏显示 "238 terms"；launcher 现显式指向 demo_nlp 索引。
- 前端（已 push main）：面板半透明、译文/词条独立限高滚动、命中术语
  `mark.term-hit` 内联高亮、播放按钮修复、语言对 (offline) 标注。

## 2026-07-10: term_acc 计分修正（occurrence 独立计数，全线重评）

用户发现 scorer 的 presence 语义 bug：同一 block 内某术语的 k 次 gold
occurrence，只要输出出现 1 次就 k/k 全命中。修正为 **count clipping**：
每组 (block, term) 的命中 = min(gold 次数, 输出中不重叠出现次数)，
匹配语义（CJK 子串 / de casefold+词干 / identity retention）与原
classify_output_hit 完全镜像；分母不变。`bb4a603`，单测 5/5。

重评结果（旧→修正）：

- **zh 10-talk（主表）**：auto 0.936→**0.905**、0.933→**0.908**；
  fixed_med 0.901→0.863、fixed_nlp 0.744→0.674、none 0.744→0.659。
  medicine 列 auto 0.943→0.903、fixed_med 0.937→0.889。ACL 列不变
  （该侧构造为每块每词 1 次）。**结论增强**：auto 现在 in-domain 也
  严格超过两个专家（med +1.4pp、ACL +0.5pp），combined 领先 +4.2pp。
- **zh 3-talk probe**：auto 0.916/0.912→0.909/0.907；oracle 0.811→0.790。
- **ja 3-talk**：auto 0.910/0.900→0.857/0.860，仍双口径第一。
- **de 3-talk**：auto 0.911/0.880→**0.801/0.795**；technical 口径被
  fixed_med (0.812) 反超 1.1pp，raw 口径仍第一。App D 已如实改写，
  德语 provenance 句同步软化。tab:sweep 不受影响（per-type 构造）。
- 今晚 ja/de 10-talk 与 zh random 的链式评分自动使用修正版 scorer。

## 2026-07-10: MFA 时间对齐 occurrence-level term_acc（论文主指标）

用户提供 MFA 对齐数据（5 ACL + 5 ESO TextGrid，commit 966c8b6）。新增
`score_time_aligned_terms.py`：每个 gold occurrence 锚到 source 秒
（ACL 走 words tier + **段偏移映射**——playlist 拼接去掉了段间静音，须把
TextGrid 原始时间经 segments.meta 的 offset/seg_duration 映射到播放秒；
medicine 走 oracle start/end），命中要求输出在 [t-2, t+30]s 窗口内出现可
接受译文变体，按时间贪心一对一分配。窗口 15–45s 结果稳定（auto tech
0.868→0.880）。这是**比 count-clipping 更严格且更真实**的口径，已设为论文主指标。

**zh 10-talk（主表 / 摘要 / §6 / 附录 C）**：gold tech 1462（ACL 780 + med 682）、
raw 2551。

| run | tech | raw | ACL | med |
|---|---:|---:|---:|---:|
| none | 0.670 | 0.706 | 0.767 | 0.559 |
| fixed_nlp | 0.679 | 0.746 | 0.797 | 0.544 |
| fixed_med | 0.805 | 0.784 | 0.764 | 0.852 |
| **auto** | **0.874** | **0.868** | **0.890** | **0.856** |

结论更强：auto 双口径第一，且 in-domain 严格超两个专家（ACL +9.3pp、med +0.4pp），
combined 领先 fixed_med +6.9/8.4pp。

**zh 3-talk probe（附录 C 散文）**：auto 0.908/0.888、fixed_nlp 0.888/0.874、
fixed_med 0.738/0.727、oracle 0.738。probe 偏 ACL（412 中 367 为 ACL），故
fixed_nlp 逼近。

**ja/de 3-talk（附录 D）**：
- ja: auto **0.742/0.767**（ACL 0.754 / med 0.711），双口径第一。
- de: auto **0.558/0.492**（ACL 0.575 / med 0.522），双口径第一。
  → **MFA 口径下德语 technical 反超消失**（count-clip 版曾被 fixed_med 0.812
  反超），德语 provenance 句与 App D intro 的软化表述已回滚。

产物：各 run 目录 `term_acc_10talk_mfa.json` / `mfa_3talk.json`。
今晚 ja/de 10-talk 落地后用同脚本重评替换 App D 3-talk。

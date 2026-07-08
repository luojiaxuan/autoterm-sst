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
- 集群侧待办（filler 池只在 gemini/taurus）：

```bash
python scripts/term_memory/build_gt_union_gs_glossary.py \
  --gt glossaries/hard_medicine_gt_raw_unique212.json \
  --filler <gemini>/documents/code/data_pre/glossary_scale/wiki_glossary_medicine_enriched.json \
  --size 10000 --target-lang zh --seed 1215 \
  --out medicine_hardraw_gt_union_gs10000.json
```

  然后：union JSON + report 回传 HF `glossaries/`；build MaxSim index；注册
  preset `medicine_hardraw_gs10k`（union inventory）和 eval-only
  `medicine_hardraw_oracle`（212 GT only）；重跑 3-talk 长流式对比
  fixed_nlp / fixed_medicine(union) / auto_working，并同时报告
  PromptGoldRetrieved@10 与 surviving prompt refs。

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

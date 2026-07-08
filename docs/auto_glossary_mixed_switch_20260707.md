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
accuracy under the same fixed prompt budget. The 4-block term accuracy run uses
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

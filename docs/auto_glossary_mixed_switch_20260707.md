# Auto Glossary Mixed-Domain Switch Benchmark - 2026-07-07

## 结论

这组结果只支持一个精确结论：在给定 clean expected domain-probe evidence 的情况下，
`auto_working` 的 target_translation_text-window state machine 可以在 ACL 5 个 talk 和
medicine 5 个 speech 组成的混合 playlist 中按 3-window consistency 规则切换
`nlp_core_10k` / `medicine_core_10k`。固定 64 windows/item 和全窗口设置都通过。

这次 benchmark 不使用 source transcript 或 ASR text。窗口文本来自 ACL 的中文 target
segments 和 RASST medicine 的中文 reference，作为 generated target translation window
的可复现实验代理。`probe_mode=expected` 表示 speech-domain probe guard 使用可控 clean
期望域证据；这验证 router state machine 和 target/probe 组合逻辑，不证明真实 speech
domain probe 的域判别质量，也不等价于完整 E2E Omni generation + real MaxSim probe
replay。`probe_mode=inverted/contested` 诊断会失败，用来确认 benchmark 能暴露错误或弱
probe 下的切换问题。

## 代码与输出

- Git ref: `c62b523 Guard probe-only glossary routing evidence`
- Taurus checkout: `/mnt/taurus/home/jiaxuanluo/rasst-demo`
- Taurus output dir:
  `/mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_c62b523`
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

## 固定 64 命令

```bash
cd /mnt/taurus/home/jiaxuanluo/rasst-demo

python3 eval/streaming_sst/eval_mixed_domain_switch.py \
  --schedule alternating \
  --windows-per-item 64 \
  --router-text-source generated_target \
  --probe-mode expected \
  --max-switch-windows 3 \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_c62b523/alternating_target64_expected_probe.json \
  --out-md /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_c62b523/alternating_target64_expected_probe.md

python3 eval/streaming_sst/eval_mixed_domain_switch.py \
  --schedule random \
  --seed 20260707 \
  --windows-per-item 64 \
  --router-text-source generated_target \
  --probe-mode expected \
  --max-switch-windows 3 \
  --out-json /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_c62b523/random_seed20260707_target64_expected_probe.json \
  --out-md /mnt/taurus/data1/jiaxuanluo/rasst_eval/auto_glossary_mixed_switch/20260707_c62b523/random_seed20260707_target64_expected_probe.md
```

## 下一步

1. 用真实 E2E server 输出的 `meta.domain_probe_scores` 和 generated target window 做 replay，
   替换 `probe_mode=expected`。
2. 如果 GPU/服务可用，直接对 ACL/medicine audio playlist 跑 streaming WS eval，记录
   active glossary over time、prompt refs、BLEU、term_ACC、masked_term_BLEU。
3. 把当前 mixed switch benchmark 纳入后续 regression：fixed 64 windows/item、
   alternating + random seed 20260707 必须通过。

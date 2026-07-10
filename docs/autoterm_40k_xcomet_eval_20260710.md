# AutoTerm vs. merged-40k xCOMET sensitivity（2026-07-10）

## 结论

在固定 5 ACL + 5 Medicine alternating playlist（16,848.115s）上，换用
xCOMET-lite 后仍没有观察到 merged-40k 的稳定翻译质量退化，也没有观察到
AutoTerm 的稳定质量增益。15s 与 30s 两种窗口的 talk-macro delta 发生变号，
且所有 paired 95% confidence intervals 都跨 0。

因此，这组 40k 结果不能支持“AutoTerm improves translation quality”。它只说明
当前规模下两个系统的整体语义质量基本持平；AutoTerm 的主要论点仍应是 modularity、
可维护性与无需用户预先选择 glossary。若继续做 merged-100k，必须先固定 protocol，
不能根据结果继续增大 glossary 直到出现预期方向。

## 评测协议

- 输入为同一 full 10-talk AutoTerm/merged streaming run。
- 根据 ACL 与 Medicine 原生句级时间戳重建相同的 source/reference 窗口，再按
  `cursor_samples` 将两个系统的 translation deltas 投到同一窗口。
- 约 15s 为主设置；约 30s 仅用于 segmentation sensitivity。
- xCOMET 使用 reference-based `{src, mt, ref}`。
- 模型：`myyycroft/XCOMET-lite`，revision
  `8d628ebffb4e3f20f53f52f9570d19dee38b9b9a`。
- 实现：NL2G/xCOMET-lite commit
  `e21e291b2a09a5a854c55b0c01c53ab580692beb`。
- 组合长度超过 480 tokens 的窗口不进入 xCOMET-lite，但仍计算 chrF2。
- Headline aggregation 为 10 个 talk 等权；95% CI 对 talk 做 20,000 次 paired
  bootstrap（seed 20260710）。

这里使用的是 278M xCOMET-lite，不是 gated 的完整 XCOMET-XL/XXL；时间窗也不是
人工语义分句，所以应将结果视为 paired sensitivity check，而不是最终论文指标。

## 结果

正 delta 表示 AutoTerm 更高。

| window / metric | eligible | Auto segment | merged segment | segment delta | talk-macro delta | talk-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 15s xCOMET-lite | 757/758 | 0.5482 | 0.5452 | +0.0030 | +0.0020 | [-0.0073, +0.0119] |
| 15s chrF2 | 758/758 | 36.7843 | 36.3924 | +0.3919 | +0.0520 | [-0.3681, +0.5062] |
| 30s xCOMET-lite | 433/451 | 0.4858 | 0.4819 | +0.0039 | -0.0014 | [-0.0164, +0.0130] |
| 30s chrF2 | 451/451 | 38.1925 | 37.7645 | +0.4280 | -0.0314 | [-0.5231, +0.4829] |

15s xCOMET-lite 的 exact talk sign-flip p=0.6992；30s 为 0.8613。按 domain
分组时，Medicine talk macro 偏 AutoTerm（15s +0.0054；30s +0.0093），NLP
偏 merged（15s -0.0014；30s -0.0122），但每个 domain 只有 5 个 talk。

作为对照，完整串接后的 corpus BLEU 为 58.2117 vs. 58.0808，
technical-masked BLEU 为 56.3912 vs. 56.0785，raw-masked BLEU 为
55.6566 vs. 55.3943，均只有很小的 AutoTerm 正差异。MFA time-aligned term
accuracy 反而是 merged 更高：technical + medicine 为 0.8550 vs. 0.8810，
raw + medicine 为 0.8103 vs. 0.8286。

实际 AutoTerm router 在 ACL 110 开头还错误切到 Education 约 4.9 分钟，完整
10-talk 的 steady-state active-domain accuracy 为 0.9813。因此当前结果同时包含
routing error 与 glossary selection，不能把差异全部归因于 glossary size。

## Source of truth 与 artifact 状态

评分代码：`eval/streaming_sst/score_xcomet_windows.py`；单元测试：
`test_score_xcomet_windows.py`。

原始 full-run staging：

```text
/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/eval_4domain_formal_20260710/full_10talk/
```

Aligned-window staging：

```text
/mnt/data1/jiaxuanluo/floras_qe_eval/work/autoterm_40k_xcomet_20260710/
```

关键 SHA-256：

```text
auto.json             a5bed299ad7b7bb83831e00c7b6ce6dcc2170afa7fcc5c5c5b0c114a206e27de
merged.json           5c72b7b76aebba58d6a7b048413f139d87300404631ca24559885b4a8c7380a1
windows_15s.jsonl     1de2650f17da69fd84a29665d4011f799ecbf45d8f2222424c1d2c5597f524b7
scored_15s.jsonl      8aea6f96d39b031f3b26814827d4a5ae565b7116462008927a5e8635e011418c
summary_15s.json      0d100127265c4c4b182a3fa4b374deddaf343865326c3608420140977a95edf8
windows_30s.jsonl     47ea0f22a089687a3d59f0155e3908988a0ba4711d0999c0bddb81cc910fecfe
scored_30s.jsonl      d3722471ae37552e5b911b2b4d8d64dc5cae5a381ad63339cb6114a5de42cc58
summary_30s.json      ff05cbd5cf796398e035f00f041aebeed84eee6ffef9c5b221e5f77fb4df3c09
```

这些 JSONL/JSON 当前仍是 Taurus local staging。它们尚未上传 Hugging Face；等
merged-100k protocol 冻结后，再与对应 catalog/eval outputs 一起选择稳定的
dataset repo，并在 Git docs 中记录 repo 与 revision。

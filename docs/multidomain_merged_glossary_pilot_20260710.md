# 4-domain AutoTerm vs. merged glossary pilot（2026-07-10）

## 目的与判定规则

这轮实验回答一个最小问题：在相同 term universe 下，自动选择当前 domain
slice 是否比始终检索 merged glossary 更好。实验前固定为 4 个 domain 各 10k，
而不是在看到结果后继续扩大 catalog 直到 merged baseline 退化。

这是 paper claim 的 pilot，不是最终主结果。本记录覆盖 3/4-talk smoke、完整
10-talk 流式运行，以及对完整运行补做的 BLEU、masked BLEU、chrF2 和
xCOMET-lite 分析；本轮不改论文正文。

## Catalog 构造

| domain | rows | 来源 |
|---|---:|---|
| NLP | 10,000 | 现有已评测 ACL union slice |
| Medicine | 10,000 | 现有已评测 medicine hard/raw union slice |
| Education | 10,000 | 8,902 exact `P31` + 1,098 strict `P31/P279` |
| Science | 10,000 | 2,389 exact `P31` + 7,611 shallow Wikipedia category paths |

- merged glossary 有 40,000 rows、39,992 unique source terms；NLP 与 Medicine
  seed slices 之间有 8 个重叠 source terms。
- Education/Science 不使用 substring 或词面关键词判 domain。每条生成项保留
  Wikidata QID/type/path，或 Wikipedia page/QID/category path。
- 这两类新增 slice 在本轮只作为真实 structured distractors；没有用
  Education/Science talk 测翻译质量，因此不能据此声称系统已在四个 domain
  上完成 translation evaluation。
- Science shallow-category slice 仍含部分跨域或 named-entity 项，后续 10-domain
  版本在入库前必须继续抽样审计和去噪。

运行时 manifest：

```text
/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/manifests/autoterm_4domain_10x4_20260710.json
```

## Router

生产路径使用最近 3 个 generated-translation chunks 的 accumulated sliding-window
context。`BAAI/bge-m3` 在 CPU 上编码窗口，并与四个双语 domain descriptions
计算 cosine similarity；keyword、routing-only MaxSim probe、speech evidence 和
metadata 仅作为辅助信号。Retriever 与 translation model 均不训练。

离线 10-way RealSI clip probe 的 119 个窗口上，base BGE-M3 description
similarity top-1 为 0.9328；结合现有 state machine 后 steady-state accuracy 为
1.0、wrong switches 为 0。真实 speech-only MaxSim probe 单独只有 4/10 top-1，
所以不能作为主 router。

## 真实流式比较

固定 playlist：

```text
ACL 2022.acl-long.268 (600s)
→ medicine_606 (600s)
→ ACL 2022.acl-long.367 (600s)
```

两个 session 在同一 Taurus server 上并发、使用相同模型与 40k term universe：

- `auto_working`：每个时刻只检索 router 选中的 10k slice；
- `merged_realsi_40k`：全程检索 40k union。

喂入间隔为每 1.92s audio chunk 等待 1.6s。两个条件均正常结束、stderr 为空。
AutoTerm 有 2/2 正确 domain transitions、0 wrong switches、steady-state domain
accuracy 1.0；切换延迟分别为 45.12s 和 32.64s。

### MFA time-aligned occurrence-level term accuracy

评分窗口为 source term 时间 `[t-2s, t_end+30s]`，同一输出 occurrence 只匹配
一次，不使用旧的“出现一次即命中同 block 全部 occurrence”计法。

| metric | AutoTerm 10k | merged 40k | delta |
|---|---:|---:|---:|
| technical + medicine | **0.9150** (377/412) | 0.9078 (374/412) | +0.72 pp |
| raw + medicine | **0.8448** (789/934) | 0.8405 (785/934) | +0.43 pp |
| NLP technical | **0.9183** | 0.9101 | +0.82 pp |
| NLP raw | **0.8425** | 0.8380 | +0.45 pp |
| Medicine | 0.8889 (40/45) | 0.8889 (40/45) | 0 |

方向符合假设，但只有 3–4 个 occurrence 的差距，尚未做 paired significance
test，不能把这轮 pilot 写成强质量结论。

### 4-talk smoke replication

为检查 3-talk 的小样本方向是否稳定，增加固定 4-block playlist：

```text
ACL 2022.acl-long.268 (600s)
→ medicine_606 (600s)
→ ACL 2022.acl-long.367 (600s)
→ medicine_605000 (600s)
```

Router 完成 3/3 方向正确的 transitions、0 wrong switches；延迟为 45.12s、
32.64s、96.96s，最后一次超过预设 60s tolerance。steady-state accuracy 为
0.9932（7 个 mismatch records），因此严格 regression gate 未通过。

| metric | AutoTerm 10k | merged 40k | delta |
|---|---:|---:|---:|
| technical + medicine | 0.9077 (403/444) | **0.9257** (411/444) | -1.80 pp |
| raw + medicine | 0.8468 (818/966) | 0.8468 (818/966) | 0 |
| NLP technical | 0.9101 | **0.9237** | -1.36 pp |
| NLP raw | **0.8425** | 0.8391 | +0.34 pp |
| Medicine | 0.8961 (69/77) | **0.9351** (72/77) | -3.90 pp |

这个 replication 与 3-talk 的微弱正增益矛盾，说明 40k 设置下的 aggregate
quality 差异仍在小样本波动范围内，不能据 3-talk 宣称 AutoTerm 优于 merged。

### Full 10-talk（4.68h）

最终运行固定交替 5 个 ACL talk 与 5 个 Medicine talk：

```text
ACL 268 → Medicine 404 → ACL 367 → Medicine 606 → ACL 590
→ Medicine 545006 → ACL 110 → Medicine 596001 → ACL 117
→ Medicine 605000
```

总音频为 16,848.115s，两个条件共享完全相同的 playlist、40k term universe
和生成设置。AutoTerm 的 router 在这条较长链上没有通过预设 gate：9 个真实
domain boundaries 中 6 个在 32 events 内切换，3 个延迟为 86.5s、309.8s 和
97.3s；steady-state active-domain accuracy 为 0.9813。ACL 110 开头还出现一次
NLP → Education 错误切换，持续约 4.9 分钟。因此下面的质量差异同时包含
glossary selection 与实际 routing error，不能把差异全部归因于 slice/merged。

#### MFA time-aligned term accuracy

| metric | AutoTerm 10k | merged 40k | delta |
|---|---:|---:|---:|
| technical + medicine | 0.8550 (1250/1462) | **0.8810** (1288/1462) | -2.60 pp |
| raw + medicine | 0.8103 (2259/2788) | **0.8286** (2310/2788) | -1.83 pp |
| NLP technical | 0.8577 | **0.9064** | -4.87 pp |
| NLP raw | 0.7968 | **0.8210** | -2.42 pp |
| Medicine | 0.8519 (581/682) | 0.8519 (581/682) | 0 |

Merged 的 term-accuracy 优势完全来自 NLP；Medicine 相同。ACL 110 的错误路由
是重要 confound，因此不能判定差异来自 glossary size；但这组实际 40k 系统结果
不支持“AutoTerm 在术语准确率上优于 merged”的表述。

#### Corpus BLEU 与 masked BLEU

| metric | AutoTerm 10k | merged 40k | delta |
|---|---:|---:|---:|
| BLEU | **58.2117** | 58.0808 | +0.1309 |
| technical-masked BLEU | **56.3912** | 56.0785 | +0.3127 |
| raw-masked BLEU | **55.6566** | 55.3943 | +0.2623 |

三个分数都把完整 4.68h hypothesis/reference 串接成一个 corpus 后计算，只能说明
merged-40k 没有造成明显的全局翻译质量下降；它们不能定位局部 hallucination、
omission 或术语提示噪声，也没有 talk-level uncertainty。

#### 对齐窗口 xCOMET-lite 与 chrF2

为补足 corpus BLEU 的粒度问题，新增
`eval/streaming_sst/score_xcomet_windows.py`：按 ACL/Medicine 原生句级时间戳
重建相同的 source/reference 窗口，再按 `cursor_samples` 将两个系统的 streaming
translation deltas 投到同一窗口。主设置为约 15s，30s 仅作为 segmentation
sensitivity；使用 reference-based `{src, mt, ref}`。模型固定为
`myyycroft/XCOMET-lite` revision
`8d628ebffb4e3f20f53f52f9570d19dee38b9b9a`，实现固定为 NL2G/xCOMET-lite
commit `e21e291b2a09a5a854c55b0c01c53ab580692beb`。组合长度超过 480 tokens
的窗口不进入 xCOMET-lite，但仍计算 chrF2。

下表同时给出 segment mean 与更保守的 talk macro。95% CI 对 10 个 talk 做
20,000 次 paired bootstrap（seed 20260710）；正 delta 表示 AutoTerm 更高。

| window / metric | eligible | Auto segment | merged segment | segment delta | talk-macro delta | talk-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 15s xCOMET-lite | 757/758 | 0.5482 | 0.5452 | +0.0030 | +0.0020 | [-0.0073, +0.0119] |
| 15s chrF2 | 758/758 | 36.7843 | 36.3924 | +0.3919 | +0.0520 | [-0.3681, +0.5062] |
| 30s xCOMET-lite | 433/451 | 0.4858 | 0.4819 | +0.0039 | -0.0014 | [-0.0164, +0.0130] |
| 30s chrF2 | 451/451 | 38.1925 | 37.7645 | +0.4280 | -0.0314 | [-0.5231, +0.4829] |

15s xCOMET-lite 的 exact talk sign-flip p=0.6992；30s 为 0.8613。窗口变长后
talk-macro delta 变号，所有 CI 均跨 0。按 domain 的方向倒是稳定：Medicine
talk macro 偏 AutoTerm（15s +0.0054；30s +0.0093），NLP 偏 merged
（15s -0.0014；30s -0.0122），但每个 domain 只有 5 个 talk，且 NLP 含上述
错误路由，不能作强结论。

这里用的是 278M xCOMET-lite，不是 gated 的完整 XCOMET-XL/XXL；15/30s
时间窗也不是人工语义分句。因此它适合作为 paired sensitivity check，不应包装成
最终论文 headline metric。它支持的结论很明确：换成比 BLEU 更语义化的指标后，
仍看不到 merged-40k 的稳定质量退化，也看不到 AutoTerm 的稳定质量增益。

### Prompt density 与 latency

| expected domain | AutoTerm refs/chunk | merged refs/chunk |
|---|---:|---:|
| NLP | 3.33 | 4.52 |
| Medicine | 2.17 | 4.03 |

Harness 只保留 reference counts，没有保留逐条 reference 内容，因此这里只能说
AutoTerm prompt 更稀疏，不能声称 irrelevant-reference precision 已提升。
Harness 报告的 `retrieve_s` p95：AutoTerm 为 335.02ms，merged 为 94.73ms；
额外 routing probes 带来
明显 latency 开销，后续需要优化或分离计时。

## 当前结论与下一步

完整 10-talk 与两种窗口的语义指标已经把 40k 结论收紧：当前不能说 AutoTerm
在固定 40k universe 上优于 zero-session-setup merged baseline。Merged 的 MFA
term accuracy 更高，而 BLEU、masked BLEU、chrF2 和 xCOMET-lite 的整体质量
基本持平。长链 router 还出现一次错误 domain 和 3 个超时切换，必须先区分
router failure 与 glossary-size effect。

后续建议：

1. 先冻结并审计 10 × 10k / merged-100k protocol，再运行同一 playlist；不要在
   看到结果后继续加到 1M 直到得到想要的方向；
2. 100k 主表同时报告 MFA term accuracy、talk-macro xCOMET/COMET、BLEU 与
   masked BLEU，并保留 paired talk uncertainty；
3. 保存逐 chunk references，增加 Gold@10、irrelevant refs/chunk 与 conflicting-term
   accuracy，直接验证“大 glossary 引入错误提示”这条机制；
4. 修复 ACL 110 的 NLP → Education 错误切换和长转场，再把 oracle-router、
   AutoTerm 与 merged 分开比较；
5. 正式 neural metric 使用完整 XCOMET-XL 或 COMET，并采用语义重分段；当前
   xCOMET-lite 结果只作 protocol smoke/sensitivity。

若预先固定的 100k 设置仍只有持平或极小增益，应把论文价值写为 modularity、
可维护性、可扩展 domain catalog 与 zero session-time setup，而不是夸大
translation quality。

## Source of truth 与 artifact 状态

- 代码分支：`explore/multidomain-routing`，核心 commits `fe313bd`、`5a8b5d8`。
- Catalog/report：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/catalog_4domain_10k_20260710/`
- MaxSim indexes：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/indexes_4domain_20260710/`
- E2E JSON、Markdown 与 MFA score：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/eval_4domain_20260710/`
- 4-talk replication 与 full 10-talk：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/eval_4domain_formal_20260710/`
- xCOMET-lite/chrF2 aligned-window staging：
  `/mnt/data1/jiaxuanluo/floras_qe_eval/work/autoterm_40k_xcomet_20260710/`
- 剩余 domain 候选采集：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/wikimedia_remaining_candidates_12500_20260710/`

关键 SHA-256：

```text
catalog_report.json                 f3a3d9adc3531349b85960db6df7cfa67f7a469a3522e68448fab7fdcc0f86d3
autoterm_4domain_10x4_20260710.json c89ee3d10f8ae6e663e7ab1be3b8f466b291f343d26f52933a943687fe158b43
index_build_report.json             2c12d640a9ee89705455e405392115df6719f1937626f755dec8bcfbf9677f6f
auto_3talk.json                     89234ba11355d3fbdb11f57af1dfb593ef5e204e9102462695214bbe7da01d28
merged_3talk.json                   4375547c21e0e5f5dd8bc3d3c82a00a1d238e58c7ce41b007e1f29cc8884cc03
mfa_auto_vs_merged_3talk.json       c1f8239fbd9a50633d689d63da92322cbe6dade63953c2c010a0be4053563c88
smoke_4talk/auto.json               741be460d4a9b63c4e259f741cabdcf9423ee45a54062f376b6e557562a7e3e8
smoke_4talk/merged.json             99f26128a0a4bd56bafa179707b5db678939077320658baf21648127d9d65846
smoke_4talk/mfa_auto_vs_merged.json 6f2ec5af0119424384fba65dad4a898e3c56e47ff33dfd599c1eba4aa93b745c
full_10talk/auto.json               a5bed299ad7b7bb83831e00c7b6ce6dcc2170afa7fcc5c5c5b0c114a206e27de
full_10talk/merged.json             5c72b7b76aebba58d6a7b048413f139d87300404631ca24559885b4a8c7380a1
full_10talk/quality_auto_vs_merged.json c562e3c98764790634fd4aaf030b43bcee90afe8b3b8fb1c53c3b1638d0c02ed
full_10talk/mfa_auto_vs_merged.json     75210da54f55d2e7bf7fa1557ef3cb8ba70abe370c206e2bb782204d58f019db
windows_15s.jsonl                   1de2650f17da69fd84a29665d4011f799ecbf45d8f2222424c1d2c5597f524b7
scored_15s.jsonl                    8aea6f96d39b031f3b26814827d4a5ae565b7116462008927a5e8635e011418c
summary_15s.json                    0d100127265c4c4b182a3fa4b374deddaf343865326c3608420140977a95edf8
windows_30s.jsonl                   47ea0f22a089687a3d59f0155e3908988a0ba4711d0999c0bddb81cc910fecfe
scored_30s.jsonl                    d3722471ae37552e5b911b2b4d8d64dc5cae5a381ad63339cb6114a5de42cc58
summary_30s.json                    ff05cbd5cf796398e035f00f041aebeed84eee6ffef9c5b221e5f77fb4df3c09
```

这些运行 JSON 与 aligned-window outputs 当前仍是 Taurus local staging，不是
canonical reusable artifact。完整 10-domain catalog 与 100k protocol 审计通过后，
将 catalog/eval outputs 整理到项目的 Hugging Face dataset（repo 尚未选定），并在
README/docs 记录 repo 与 revision；在此之前不要删除上述 staging 路径。

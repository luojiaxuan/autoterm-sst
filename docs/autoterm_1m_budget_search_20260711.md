# AutoTerm active-budget 与 merged glossary 正式比较（2026-07-11）

## 结论

本轮在同一条完整四 talk 流上比较 Known-domain、AutoTerm、Merged-100k 和
Merged-1M。结果支持一个有边界的系统结论：100k merged glossary 仍在当前
retriever 的承受范围内，甚至取得最高 TERM_ACC；扩展到 1M 后，固定候选与
prompt evidence budget 开始失效，TERM_ACC、BLEU 和 masked BLEU 同时下降。
AutoTerm 只激活至多 4k terms，却保持与 Known-domain 几乎相同的 TERM_ACC，
并显著提高 prompt precision。

因此论文不声称“大 glossary 必然更差”，而是强调：RASST 的 glossary RAG
解决了 OOD term exposure，但 universal glossary 会随 catalog 扩张持续增加
ranking/prompt noise。AutoTerm 在其上增加一个无需 session-time domain input 的
budgeted multi-slice working set。

## 冻结设置

- Playlist：ACL 268、Medicine 545006、ACL 367、Medicine 606；总长
  5,381.508 秒。
- 共同 decoder timing：2,803 个完整 windows；四个条件逐 window 相同。
- Known-domain-2.5k：已知 talk domain，激活对应 2.5k slice。
- AutoTerm-1k×4：10 个 1k slices，最近 4 个 streaming translation chunks 与
  bilingual prototypes 做 BGE-M3 cosine similarity，激活 semantic top-4。
- Merged-100k：Merged-1M 的严格嵌套前缀，89,216 个 target-bearing mappings
  加 10,784 个 general distractors。
- Merged-1M：同一 target-bearing prefix 加 general distractors 到 1M。
- 所有条件：retrieval candidate budget 100、MaxSim threshold 0.78、prompt top-10。
- 跨 domain 的同 source、不同 translation mappings 全部保留；只移除完全相同的
  source-target pair duplicates。

## TERM_ACC scorer 修复

旧 gold builder 对每个 glossary source term 使用轻量单复数容错，因此同一个
spoken token 会同时生成 `model` 和 `models` 两条 annotation。旧 scorer 又按
source term 建独立 one-to-one pool，使同一译文可以重复得分。1079 raw annotation
rows 中有 231 条属于这种 alias excess。

commit `7dc4ac0` 将 headline occurrence 定义修正为：仅当 domain、block、精确
audio span 和 NFKC/space/case-normalized target variant set 全部相同时，合并
source aliases。不同 target variants、嵌套术语和同一 source 的 domain-dependent
translations 仍分别计数。原始 source aliases 继续用于 prompt precision。

- Raw headline：1,079 annotation rows → 848 occurrences（615 NLP，233 Medicine）
- Technical：630 annotation rows → 561 occurrences
- Headline fingerprint：
  `aedee99367569f6b1b92b47d99259bf9678ed792a86404f5f9b831e5af80498c`

这也修正了旧结果中 AutoTerm 表面上高于 Known-domain 的异常。新结果为
Known-domain 724 hits、AutoTerm 723 hits；两者只差 1/848。

## 完整四 talk 正式结果

| Setting | TERM_ACC | NLP | Medicine | BLEU | T-MBLEU | R-MBLEU | Prompt precision | Refs/chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Known-domain-2.5k | 85.38 (724/848) | 86.67 | 81.97 | 53.77 | 51.01 | 50.20 | 39.39 | 1.12 |
| AutoTerm-1k×4 | 85.26 (723/848) | 87.32 | 79.83 | **54.03** | **51.26** | **50.46** | 39.29 | 1.12 |
| Merged-100k | **86.32 (732/848)** | **88.46** | 80.69 | 53.94 | 51.16 | 50.41 | 8.83 | 5.00 |
| Merged-1M | 83.02 (704/848) | 86.67 | 73.39 | 52.12 | 49.36 | 48.59 | 4.37 | 8.79 |

关键解释：

- Merged-100k 的 TERM_ACC 最好，说明 100k 仍是可信、可承受的 baseline；论文
  不应把它描述为失败。
- 100k 已需要约 4.5 倍于 AutoTerm 的 references/chunk，prompt precision 从
  39.29% 降到 8.83%，但整体质量尚未明显下降。
- 1M 时 prompt precision 进一步降到 4.37%，TERM_ACC 相对 AutoTerm 低 2.24
  points，BLEU / T-MBLEU / R-MBLEU 分别低 1.91 / 1.90 / 1.88。
- AutoTerm 与 Known-domain TERM_ACC 相差 0.12 points；不能声称显著优于
  Known-domain，应使用 `matches` 或 `within 0.12 points`。

## Selected-window tuning 协议

参数搜索只使用固定四窗口、570 秒的开发协议，不能作为论文 headline。scorer
修复后该协议版本化为 `frozen4_selected_window_v2`：179 raw rows → 142
alias-dedup occurrences，fingerprint 为
`4589ea5ebb87e4fb0cf59cfd4888034d46db547fbe9dbbde53dc9121a59e90c1`。

冻结的三系统 tuning 复算如下：

| Setting | TERM_ACC | BLEU | T-MBLEU | R-MBLEU | Prompt precision | Refs/chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Known-domain-2.5k | 77.46 (110/142) | 45.46 | 42.73 | 40.98 | 47.57 | 1.52 |
| AutoTerm-1k×4 | 77.46 (110/142) | 45.53 | 43.13 | 41.84 | 51.94 | 1.30 |
| Merged-1M | 75.35 (107/142) | 42.78 | 40.91 | 39.10 | 7.35 | 8.97 |

## Router 机制与边界

当前路由不是 source transcript classifier。它对最近四个 streaming target chunks
与每个 slice 的 description/bilingual prototypes 做 semantic similarity，并在总
active-term budget 内选择 top-K slices。top-4 的作用是缓解 top-1 domain lag：完整
流上 correct-slice inclusion 为 98.64%，而 top-1 active-domain accuracy 为 96.97%。
三次 domain boundary 的 top-1 switch 仍需 40.4 / 43.5 / 82.1 秒，这是依赖
generated target context 的已知限制。

## 10-talk 扩展协议与运行状态

论文主表正在扩展到一条完整的 10-talk alternating stream：

```text
ACL 268 → Medicine 404 → ACL 367 → Medicine 606 → ACL 590
→ Medicine 545006 → ACL 110 → Medicine 596001 → ACL 117
→ Medicine 605000
```

- 总音频为 269,569,844 samples / 16,848.115 秒；chunk 为 30,720 samples，
  每个条件必须精确产生 8,776 个 decoder windows。
- ACL metadata 共有 468 个 segment。两台执行机器只重写旧 Taurus 绝对路径的
  prefix，WAV 顺序和内容不变；canonical playlist SHA-256 为
  `6c8d08949efd91a843fbf6c13f2fa0196f848242742709023c51792c6c12a6e7`。
- headline TERM_ACC 固定为 2,047 个 alias-dedup occurrences（NLP 1,368，
  medicine 679），来自 2,551 个 raw annotation rows；gold fingerprint 为
  `a5513f8194cab5378ab95fab0ed386d88f36cc676c5dfde900f55dfc1c6b1b69`。
- 四个条件为 Known-domain-1k、AutoTerm-1k×4、Merged-100k、Merged-1M。
  Known-domain 直接选择 AutoTerm 使用的同一个正确 1k slice；每个 1k slice
  已覆盖对应 domain 的全部 gold source-target pairs，2.5k 只会增加 distractors。
- 这不是 held-out evaluation。NLP 1k 以 ACL technical glossary 为前缀，medicine
  1k 也包含五个 medicine talks 的人工审核术语。该实验隔离的是 routing、retrieval
  ranking 和固定 prompt budget 的行为，不评估 glossary induction/generalization。

正式运行采用 exact code commit `3b7a39a`，并强制以下 gate：10 blocks、全部音频
路径存在、8,776 windows、首批 retrieval 的 scored inventory 分别为 1k / 4k /
100k / 1M、四条件 timing signature 完全一致，以及 raw denominator 精确为 2,047。
commit `4a6449e` 又给 mixed-audio evaluator 增加了 fail-fast 输入校验，后续请求的
ACL/medicine item 未全部加载时会在启动网络/GPU 评测前失败。

截至本记录，六条 audited run（B200 上 Known/Auto，Hyper00 上四条件，其中
Known/Auto 是冗余同硬件复跑）仍在运行。所有更早的 missing-import、重复 client、
错误 preset 和 medicine-only 尝试均已隔离为 `.invalid_*` 或独立 smoke 文件，不能
进入 scorer。正式 local staging 根目录为：

```text
B200:   /data02/jaxan/autoterm-10talk-budget-20260711/b200/
Hyper:  /data02/jaxan/autoterm-10talk-budget-20260711/hyper/
```

最终聚合 JSON/scorecard/timeline 应上传到稳定的 Hugging Face dataset；repo 与
revision 当前仍为 TBD，以上路径只是 active local staging。

## Source of Truth 与 artifact status

| 内容 | 位置 / SHA-256 | 状态 |
| --- | --- | --- |
| scorer/router code | GitHub `luojiaxuan/autoterm-sst`, branch `explore/multidomain-routing`, commit `7dc4ac0` | 已 commit 并 push |
| Merged-100k glossary | Hyper00 `.../merged_pair_general_100k/`；glossary `31bf1e79...a8`，index `fdb214be...74ec` | local staging；HF pending |
| Known-domain run | `full4_oracle2p5k_same_hyper_edd7.json`；`d21817fe...154f5` | local staging；HF pending |
| AutoTerm run | `full4_auto1k_k4_w4_bilingual_edd7.json`；`d7ccd1ba...9b39c` | local staging；HF pending |
| Merged-100k run | `full4_merged100k_nested_edd7.json`；`8c8dfe19...0c19` | local staging；HF pending |
| Merged-1M run | `full4_merged1m_edd7.json`；`16ee87fa...7d8` | local staging；HF pending |
| 100k scorecard | `full4_same_hyper_story_100k_alias_dedup_score.json`；`ac8e8cc8...3d0` | local staging；HF pending |
| 1M scorecard | `full4_same_hyper_story_1m_alias_dedup_score.json`；`91e405f3...eaa` | local staging；HF pending |

所有正式 run 使用相同 playlist SHA
`a91adef64273f210b28d898b4f9b91bf30ae006e6c363aee079c2a79258ae40e`
和 timing SHA
`cf8f28cb400166af7fc950dd798939ff279828f0af9f5ff7e8a9d64e25834ac7`。

这些 JSON、glossary 和 indexes 目前仍是 Hyper local staging，不是 canonical
release artifact。论文 release 前应上传聚合 artifacts 到稳定 Hugging Face dataset
repo，并将 repo URL 与 revision 回填到本页和 README。

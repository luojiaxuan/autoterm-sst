# AutoTerm 十 talk 主结果与 StreamLAAL 指标真源（2026-07-11）

## 结论

同一条 5 个 ACL talk 与 5 个 medicine talk 交替流支持如下系统叙事：

1. InfiniSST no-RAG 提供一个系统级 no-retrieval reference，但它与 RAG 条件使用
   不同 base model，因此不能把全部差值解释为 controlled retrieval ablation。
2. AutoTerm-1k×4 不需要 session-time domain/glossary input，但与已知正确 domain
   的 Known-domain-1k 基本持平。
3. Merged-100k 仍是可用的中间 baseline；扩展到 Merged-1M 后，固定 retrieval 与
   prompt budget 中的无关 evidence 增多，TERM_ACC、BLEU 和 MT-BLEU 同时下降。

因此论文不声称“大 glossary 必然失败”，也不把 100k 作为普适阈值。核心贡献是
用 recent streaming translation context 管理一个 budgeted multi-slice working set，
在 catalog 持续扩展时避免让用户上传 glossary 或选择 domain。

## 唯一可用于论文的指标协议

- **TERM_ACC**：固定 2,047 个 raw tagged occurrences（NLP 1,368；medicine
  679）。2,551 个 raw annotation rows 只在 domain、block、精确 audio span 和
  normalized target variants 完全相同时合并 source aliases；不同 target、嵌套术语
  和跨 domain translation 均保留。所有条件使用同一分母。
- **BLEU**：每个 talk 的 streaming hypothesis 先经过 RASST/StreamLAAL 的
  `mwerSegmenter` 对齐到 reference sentences，再在全部 1,905 个句子上计算 corpus
  sacreBLEU，不能平均 ACL/medicine 两个 BLEU，也不能对整段字符串直接算 BLEU。
- **MT-BLEU**：使用同一 StreamLAAL/mWER pipeline，并分别按 ACL raw glossary 与
  medicine raw glossary 屏蔽 target translations。不报告基于 UI 高亮
  technical/display subset 的第二个 masked-BLEU。
- **Prompt precision**：注入 prompt 的 references 中，与当前 time-local raw source
  occurrence 重叠的比例；**Refs/chunk** 是每个 decoder window 实际注入条数均值。

旧四-talk scorecard 中的 concatenated BLEU 和两种非 canonical masked variants
全部为已废弃探索指标，不能再进入论文、README 或后续表格。

## 正式设置

- Playlist：5 ACL + 5 medicine complete talks，交替排列；约 4.7 小时。
- Chunk/stride：1.92 秒；每个 RAG 条件精确 8,776 个 decoder windows，timing
  signature 相同：
  `7c0be6da5557cb99b3276967f3e575c353a03515e68d64489bffada22b2dfa7c`。
- Known-domain-1k：使用 talk 的正确 1k slice，需要 domain input。
- AutoTerm-1k×4：十个 1k topic slices；最近四个 generated target chunks 与
  bilingual prototypes 做 BGE-M3 cosine similarity；最多激活四个 slices / 4k terms。
- Merged-100k 与 Merged-1M：严格前缀嵌套 universal glossaries。仅去除完全相同的
  source-target pair；跨 domain 的同 source、不同 translation mappings 保留。
- 所有 RAG 条件：100 retrieval candidates、MaxSim threshold 0.78、prompt top-10。
- 这是 controlled routing/retrieval-competition evaluation：NLP 与 medicine 1k
  slices 包含 benchmark terminology，不是 held-out glossary induction。

## 正式结果

机器可读版本：[`autoterm_10talk_streamlaal_20260711.tsv`](autoterm_10talk_streamlaal_20260711.tsv)。

| Setting | TERM_ACC | NLP | Medicine | BLEU | MT-BLEU | Prompt precision | Refs/chunk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| InfiniSST (no RAG) | 51.10 (1,046/2,047) | 50.22 | 52.87 | 39.26 | 36.70 | 0.00 | 0.00 |
| Known-domain-1k | 88.08 (1,803/2,047) | 88.82 | **86.60** | **45.19** | **42.39** | **48.68** | **0.69** |
| AutoTerm-1k×4 | **88.32 (1,808/2,047)** | 89.33 | 86.30 | 45.09 | 42.28 | 34.78 | 0.94 |
| Merged-100k | 87.79 (1,797/2,047) | **89.40** | 84.54 | 43.72 | 40.93 | 6.85 | 4.96 |
| Merged-1M | 84.42 (1,728/2,047) | 87.13 | 78.94 | 42.79 | 39.95 | 3.44 | 8.70 |

关键差值：

- AutoTerm vs no RAG：+37.22 TERM_ACC points、+5.83 BLEU、+5.58 MT-BLEU。
- AutoTerm vs Known-domain：+0.24 TERM_ACC points、-0.10 BLEU、-0.11 MT-BLEU；
  论文只能写 `matches`，不能写严格优于 oracle/reference。
- AutoTerm vs Merged-100k：+0.54 TERM_ACC、+1.37 BLEU、+1.36 MT-BLEU；100k
  在 NLP 仍略好，但 medicine 已落后 1.76 points。
- AutoTerm vs Merged-1M：+3.91 TERM_ACC、+2.30 BLEU、+2.33 MT-BLEU；prompt
  precision 为 34.78% vs 3.44%，Refs/chunk 为 0.94 vs 8.70。

MT-BLEU 已屏蔽被评测术语本身，AutoTerm 对 1M 仍有 2.33 points 优势，说明无关
prompt evidence 不只影响 tagged-term copy，也会影响周边翻译质量。

## No-RAG baseline provenance

InfiniSST hypotheses 来自 `LeiLiLab/RASST` PR #1：
`agent/add-masked-bleu-results`，commit
`e0682eaa094c00004192cae0045fae8b6ffacca4`，使用与主实验相同的十 talks 与
1.92 秒 policy。这里没有直接平均 PR 中 ACL/medicine 的旧表格，而是把 hypotheses
重新 materialize 到当前共同 references，用同一 mWER、MT-BLEU 和 2,047-occurrence
TERM scorer 重算。No-RAG prompt precision 与 Refs/chunk 按定义为 0。
该 row 使用 InfiniSST，而四个 RAG rows 使用 Qwen3-Omni；论文只能称其为系统级
reference，不能用它单独证明 retrieval 的因果增益。

## Router 与错误分析

- 同一 AutoTerm run 中，正确 slice 在 four-slice working set 内的覆盖率为
  96.93%（8,507/8,776 windows）；Figure 3 只展示 talk domain、selected slice 和
  correct-slice inclusion，不展示内部 talk ID 或重复 metric cards。
- ACL→medicine 边界后，旧 context 曾让 medicine slice 短暂缺席，随后进入 working
  set。这是 generated-target causal routing 的已知边界。
- 在一个讨论 scientific papers 的 medicine 段落，Merged-1M 同时检索出
  `paper→论文` 与 `paper→纸`，AutoTerm 只保留 context-appropriate mapping。该例
  说明 routing 减少但不能完全消除 ambiguous evidence。

## Source of Truth 与 artifact status

| 内容 | 位置 / SHA-256 | 状态 |
| --- | --- | --- |
| scorer / bundle code | GitHub `luojiaxuan/autoterm-sst`, branch `explore/multidomain-routing` | 已 commit/push；以本页所在 revision 为准 |
| no-RAG run | Taurus `.../runs/norag.json`; `ff68d613...f9b2a4` | local staging；HF pending |
| Known run | `8104be3d...8e93` | complete local staging；HF pending |
| AutoTerm run | `3b4fbb01...32b30f` | complete local staging；HF pending |
| Merged-100k run | `dd03b4f7...3b58e` | complete local staging；HF pending |
| Merged-1M run | `a3ad7235...9a0cee` | complete local staging；HF pending |
| fixed TERM score | `scores/mfa_term_all.json`; `e46a5aa5...6d219d` | complete local staging；HF pending |
| common source / reference | `63783f4a...159a0` / `1c1477dc...745880` | 1,905 aligned sentences；HF pending |
| raw MT mask glossary | `1f093f79...d276a` | 450 stored mappings; per-domain masking in scorer |
| mWERSegmenter | `09da1798...57a157` | pinned executable |

Taurus staging root：
`/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/`。

完整 runs、bundles、per-sentence outputs 和 score JSON 尚不是 canonical release
artifact。预定 Hugging Face dataset repo：
`luojiaxuan/autoterm-sst-10talk-streamlaal-zh`，当前状态 **upload pending**。上传后需
在本页与顶层 README 回填 repo URL、revision/tag、schema 和生成命令。

## 复现入口

- bundle/source materialization：`eval/streaming_sst/materialize_mixed_streamlaal.py`
- StreamLAAL BLEU：`eval/streaming_sst/score_streamlaal.sh`
- per-domain raw MT-BLEU：`eval/streaming_sst/score_resegmented_masked_bleu.py`
- fixed-denominator TERM_ACC：`eval/streaming_sst/score_time_aligned_terms.py`
- Figure 3：`eval/streaming_sst/render_budgeted_routing_timeline.py`

当前校验：materializer 与 MT-BLEU 单元测试通过；五个 conditions 共用 source、
reference、mask glossary 和 1,905-sentence denominator。

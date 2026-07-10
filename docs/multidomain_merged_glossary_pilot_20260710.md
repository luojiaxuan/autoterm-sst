# 4-domain AutoTerm vs. merged glossary pilot（2026-07-10）

## 目的与判定规则

这轮实验先回答一个最小问题：在相同 term universe 下，自动选择当前
domain slice 是否比始终检索 merged glossary 更好。实验前固定为 4 个 domain
各 10k，而不是在看到结果后继续扩大 catalog 直到 merged baseline 退化。

这是 paper claim 的 pilot，不是最终主结果。只有方向明确后才值得继续完成
10-domain × 10k；本轮不改论文正文。

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

这轮结果达到最小 go 条件：在固定 40k universe 上，AutoTerm 的两个 MFA 指标
均严格高于 zero-session-setup merged baseline，并保持正确在线切换。但增益很小，
论文要站得更稳仍需：

1. 完成并审计剩余 6 个 domain，形成预注册的 10 × 10k / merged-100k 设置；
2. 在同一 3–4 talk playlist 上重复 AutoTerm vs merged，报告 paired occurrence
   差异，而不是只看单次 aggregate；
3. 保存逐 chunk references，增加 Gold@10、irrelevant refs/chunk 与 conflicting-term
   accuracy；
4. 优化 domain-probe 开销，再报告拆分后的 active retrieval、router 和 generation
   latency。

若 100k 仍只有持平或极小增益，应把论文价值写为 modularity、可维护性与
zero session-time setup，而不是夸大 translation quality。

## Source of truth 与 artifact 状态

- 代码分支：`explore/multidomain-routing`，核心 commits `fe313bd`、`5a8b5d8`。
- Catalog/report：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/catalog_4domain_10k_20260710/`
- MaxSim indexes：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/indexes_4domain_20260710/`
- E2E JSON、Markdown 与 MFA score：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/eval_4domain_20260710/`
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
```

这些大文件当前仍是 Taurus local staging，不是 canonical reusable artifact。
完整 10-domain catalog 审计通过后再上传 Hugging Face，并在 README/docs 记录 repo
与 revision；在此之前不要删除上述 staging 路径。

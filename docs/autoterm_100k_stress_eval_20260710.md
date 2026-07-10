# 10-domain 100k glossary stress validation（2026-07-10）

## 结论

这轮 3-talk 结果**不支持**“AutoTerm 的端到端质量优于 merged-100k”。在同一
ACL 268 → Medicine 545006 → ACL 367 playlist 上，AutoTerm top-4/40k working
set 的 MFA time-aligned term accuracy 比 raw merged-100k 低约 4.6 个百分点，
BLEU 低 0.60，technical/raw masked BLEU 也分别低 0.52/0.19。

AutoTerm 的正向结果是 prompt 更稀疏、time-local precision 更高：refs/chunk 从
5.82 降至 3.13，technical/raw prompt precision 分别提高 1.55/1.61 个百分点。
但这不足以抵消 NLP slice 经常缺席 active working set 的问题，也不能据此宣称
translation quality gain。

当前最重要的 router failure mode 是：表面 steady-state active-domain accuracy
为 0.9740，但 expected slice coverage 只有 0.8541；NLP slice 仅出现在 0.7285
的 NLP output events 中，Medicine slice coverage 为 0.9858。因此只报告
`active_domain` 会高估实际 glossary routing 质量。

## 固定协议

- Playlist：ACL `2022.acl-long.268` → Medicine `545006` → ACL
  `2022.acl-long.367`，总音频 2,538.689 秒。
- Language pair：English → Chinese；chunk 1.92 秒。
- AutoTerm：10 个 10k domain slices，context-similarity-only routing，最多选择
  4 个 slices（40k terms），全局 prompt cap 仍为 10。
- Merged：`merged_realsi_100k`，100,000 rows，88,622 unique normalized source
  terms；直接检索整个 union。
- Headline term metric：MFA time-aligned occurrence-level TERM_ACC，同一份 432
  technical+medicine / 881 raw+medicine gold occurrences，一次输出不能命中多个
  occurrence。
- 质量指标：corpus BLEU、technical-masked BLEU、raw-masked BLEU。
- Prompt 指标：每个实际注入 decoder prompt 的 reference 都按当前 MFA source
  occurrence window 判断相关性。

AutoTerm retry 使用 cursor acknowledgement backpressure（最多 1 个未确认 chunk；
30 秒 stall timeout），merged pilot 使用 `feed_sleep=1.0`。两者 playlist 和 audio
chunking 相同，但 persisted output event 数为 1,295 vs. 1,037，且 AutoTerm 出现
28 次 timeout release。因此这是快速 falsification / stress result，不是可做
paired significance claim 的最终 protocol。

## Catalog 构造与适用范围

10 个 slices 为 NLP、Medicine、Education、Finance、Legal、Environment、
Entertainment、Science、Sports、Art，每个 10,000 rows。NLP/Medicine 使用现有
评测资源；其余 domains 使用 Wikidata P31/P279 路径或 Wikipedia category/deep
category 采集，**没有用 substring 判 domain**。

100k rows 中有 88,622 unique source terms、9,802 个跨 slice 重叠 source term
types，以及 594 个多 target-variant term types。额外八个 slices 是 catalog-scale
stress distractors；本实验没有对应八个 domain 的 talk/reference，不能声称系统
已在十个 domain 上通过 translation evaluation。

## 结果

### MFA time-aligned TERM_ACC

| metric | AutoTerm top-4/40k | raw merged-100k | AutoTerm − merged |
|---|---:|---:|---:|
| Technical + Medicine | 0.8773 (379/432) | **0.9236** (399/432) | -0.0463 |
| Raw + Medicine | 0.8581 (756/881) | **0.9047** (797/881) | -0.0466 |
| NLP technical | 0.8734 | **0.9266** | -0.0532 |
| NLP raw | 0.8555 | **0.9052** | -0.0497 |
| Medicine | **0.9189** (34/37) | 0.8919 (33/37) | +0.0270 |

Medicine 小分组只有 37 occurrences，不能把 +1 hit 解释成稳定优势。整体下降主要
来自 NLP。

### Translation quality

| metric | AutoTerm top-4/40k | raw merged-100k | AutoTerm − merged |
|---|---:|---:|---:|
| BLEU | 54.3763 | **54.9718** | -0.5955 |
| Technical-masked BLEU | 51.9039 | **52.4203** | -0.5164 |
| Raw-masked BLEU | 50.5896 | **50.7813** | -0.1917 |

masked BLEU 没有反转结论：merged glossary 在这三个 talks 上没有表现出更大的
非术语翻译质量损失。

### Prompt precision 与 retrieval cost

| metric | AutoTerm top-4/40k | raw merged-100k | AutoTerm − merged |
|---|---:|---:|---:|
| Technical prompt precision | **0.094898** | 0.079357 | +0.015541 |
| Raw prompt precision | **0.167365** | 0.151259 | +0.016106 |
| Retrieved refs/chunk | **3.132819** | 5.820636 | -2.687817 |
| Retrieval p50 | 89.68 ms | **64.74 ms** | +24.94 ms |
| Retrieval p95 | 2,203.98 ms | **208.57 ms** | +1,995.41 ms |

AutoTerm 插入更少、更精确的 references，但四个 index 的长尾 latency 明显更差。
最后 100 个 events 的 retrieval p50/p95 为 163.07/4,811.08 ms，最大值
11,899.29 ms；这需要在扩大正式实验前 debug，而不是作为可接受的 demo latency。

## Router 与 working-set 诊断

- `NLP → Medicine` active-domain switch latency：38.517 秒，19 events，pass。
- `Medicine → NLP` active-domain switch latency：124.184 秒，64 events，fail。
- wrong switches：0；steady-state active-domain accuracy：0.9740。
- Medicine slice 在第一个边界后 0.117 秒进入 top-4；NLP slice 在第二个边界后
  18.584 秒进入 top-4。
- Expected slice coverage：overall 0.8541，NLP 0.7285，Medicine 0.9858。
- 按 reference 自报 source domain 统计，63.36% 的 NLP refs 和 57.33% 的
  Medicine refs 来自当前 expected domain；其余来自其他三个 active slices。

这说明 top-4 policy 比 hard top-1 更能提前覆盖新 domain，但其 domain-description
similarity 排名不足以保证正确 slice 常驻。下一步应该先加入“当前 active slice
必须保留”或 calibrated inclusion gate，并用 router replay 检验 expected-slice
coverage；不应继续扩大 catalog 直到 merged baseline 自然变差。

## 运行完整性与失败记录

成功 retry 共处理 1,323 server batches，`generation_ok=1` 为 1,323/1,323。
Backpressure 最终确认全部 40,619,016 samples，但发生 28 次 30 秒 timeout release，
最大观察到 3 个未确认 chunks。该 pacing 异常也应在正式 rerun 前修正。

此前一次 formal attempt 在 server batch 847（其中前 32 batches 为 smoke，即正式
run 约 815 chunks）发生 `Engine core proc EngineCore_DP0 died unexpectedly`；随后
batch `generation_ok=0`，未产生有效结果 JSON。该失败 run 只作为诊断日志保留，
不参与任何分数。

## Source of truth 与 artifact 状态

- Runtime/evaluator/scorer code：`explore/multidomain-routing` commit
  `89f4fc4198d3cad721d75facd151b95516c0d6ab`。
- Git lightweight summary：
  `runtime/eval_20260710/autoterm_100k_3talk_summary.json`。
- AutoTerm raw output staging：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/formal3_20260710/autoterm_top4_bp1_retry.json`。
- Raw merged output staging：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/search_4talk_20260710/pilot3_raw_merged100k.json`。
- Catalog report staging：
  `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/catalog_10domain_100k_deepcat_20260710/catalog_report.json`。
- Canonical input data：Hugging Face `gavinlaw/rasst-main-result-data`, revision
  `204ba141` for the compact medicine GT noted in `docs/internal_operations.md`。

关键 SHA-256 已写入 lightweight summary。raw streaming JSON、generated catalog、
MaxSim indexes 与评分明细当前仍是 Taurus local staging，尚未上传 Hugging Face；
不能把这些本地路径当作可复用 canonical artifacts。全球 source-dedup + top-up 的
100k baseline 已构建但尚未运行，因此本页必须标记为 provisional raw-union
comparison。

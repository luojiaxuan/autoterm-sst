# 十 talk 正式结果的补充证据(2026-10-24)

本页给 `docs/autoterm_1m_budget_search_20260711.md` 的十 talk 正式 run 补三类证据:配对显著性、无术语句的译文质量、以及路由的 open-loop replay。全部只用已审计的输出,不重新解码。脚本在 `eval/streaming_sst/evidence/`,结果 JSON 在本目录。

## 输入

- 四个 RAG run(Known-domain-1k、AutoTerm-1k×4、Merged-100k、Merged-1M)的 run JSON、bundle 与 `scores/*/segments.jsonl`,来自 Taurus `/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/`;AutoTerm run SHA-256 `3b4fbb01c8d120c432f595bd788b950f263da93105c45e5a32ae9caba632b30f`。
- 术语 gold:`eval/streaming_sst/mfa_alignments`、`acl_gold_technical.json`、`acl6060_tagged_gt_raw_min_norm2.json`、medicine oracle term maps;MT-BLEU 遮罩用 ACL raw glossary 与 `hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json`。
- 复现核对:逐出现打分后四个 run 的命中数为 1803 / 1808 / 1797 / 1728(与正式表一致);重算 MT-BLEU 为 42.39 / 42.28 / 40.93 / 39.95(一致)。

## 1. 配对显著性(`paired_significance.py` → `paired_significance.json`)

主推断以 talk 为单位:精确 talk 级标签互换置换检验(2^10 种组合,每种重算合并指标;10 个 talk 下可达到的最小双侧 p 是 2/1024 ≈ 0.002),以及按领域分层(每次抽 5 个 ACL + 5 个 ESO talk)的 talk 级 bootstrap 95% CI。BLEU 一律由合并的 sacreBLEU 充分统计量按 sacreBLEU 自身的打分(含 exp 平滑)计算。逐出现 McNemar 与句级 bootstrap 只作敏感性分析(它们把相关的观测当独立样本)。

| 对比 | ΔTERM_ACC(点) | ΔBLEU | ΔMT-BLEU |
|---|---|---|---|
| AutoTerm − Known | +0.24,CI [−0.95, +1.49],p=0.86 | −0.10,CI [−0.50, +0.38],p=0.70 | −0.11,CI [−0.57, +0.39],p=0.73 |
| AutoTerm − Merged-100k | +0.54,CI [−0.60, +1.84],p=0.48 | +1.37,CI [+0.86, +1.80],p=0.004 | +1.36,CI [+0.65, +1.81],p=0.018 |
| AutoTerm − Merged-1M | +3.91,CI [+2.03, +5.76],p=0.012 | +2.30,CI [+1.71, +2.95],p=0.002 | +2.33,CI [+1.10, +3.06],p=0.002 |

- 去掉任意一个 talk,AutoTerm − 1M 三项差值都保持为正(TERM ≥ +3.28,BLEU ≥ +2.00,MT-BLEU ≥ +1.99);AutoTerm − Known 的去一 talk 区间都跨 0 附近(TERM [−0.17, +0.77])。
- Holm 校正若把表中 12 个对比(另加 Known − 1M)当作一个族,AutoTerm − 1M 的 TERM_ACC 校正后 p=0.082,BLEU / MT-BLEU 为 0.023;只把 "AutoTerm 优于 1M" 的三个指标当作一个族时三项校正后都 < 0.05。本分析没有预注册主终点,措辞以 CI 为准。
- 读法:与 Known-domain 相比,在这 10 个 talk 上没有检测到差异,只能写 "no statistically detected difference",不能写 equivalent;等价性需要事先固定的 margin 和新数据。对 1M 的优势在 talk 级推断下成立。
- 不覆盖的东西:没有新 talk、没有不同播放顺序或解码随机性;路由配置曾在这些 talk 的 4 个窗口上选过。

## 2. 无术语句的译文质量(`term_free_quality.py` → `term_free_quality.json`)

子集按参考译文定义:参考里不含任何 glossary 目标串的句子(935 / 1,905),不按任何系统的术语结果分组,也不需要遮罩。

| 对比 | ΔBLEU(无术语句) | 分层 talk bootstrap CI | talk 置换 p | ACL / medicine |
|---|---|---|---|---|
| AutoTerm − Merged-1M | +2.19 | [+0.66, +3.46] | 0.023 | +0.19 / +2.23 |
| AutoTerm − Merged-100k | +1.69 | [+1.08, +2.16] | 0.012 | −0.17 / +1.70 |
| Known − Merged-1M | +2.31 | [+1.07, +3.35] | 0.016 | +2.03 / +2.35 |
| AutoTerm − Known | −0.11 | [−0.76, +0.35] | 0.80 | −1.84 / −0.12 |

读法:大 merged glossary 在没有术语可查的句子上也更差,差异主要来自 medicine talk。"参考里没有 glossary 串" 不等于 "没有术语",也不能证明注入的条目都无关;要把这个差异归因于 prompt 噪声,需要固定前缀、只改无关条目的干预实验。

此前按 "哪个系统译对的术语更多" 分组再遮罩术语算 BLEU 的做法已弃用:分组以系统输出为条件,且遮罩只去掉正确术语串,错误译法留在对手的输出里会单方面拉低其分数。

## 3. 路由 open-loop replay(`router_replay.py`、`rapid_switch_replay.py`)

把正式 AutoTerm run 的生成译文固定,窗口 t 用之前 w 个 chunk 的译文做 context,按 `framework/agents/term_memory/context_similarity.py` 的方式编码(BGE-M3 CLS、128 token、左截断),与 slice prototype 算余弦并排序。不含线上 router 的平滑、切换门槛与 pin,所以是裸相似度的诊断,不能把它与线上结果的差距归给某个单独部件。prototype 由 manifest description + `DOMAIN_ROUTER_PROTOTYPES` 重建(1k slice 的 manifest 原件不在可访问的盘上,description 取同一 10k manifest)。

裸排序(原始 prototype):

| w(chunk) | top-1 | 前 4 覆盖 | top-1 变化次数 / 8,776 窗口 |
|---|---|---|---|
| 1 | 36.6% | 49.5% | 6,530 |
| 2 | 54.4% | 68.0% | 3,889 |
| 4 | 72.6% | 85.0% | 1,809 |
| 8 | 84.5% | 94.8% | 792 |

线上 router(w=4,领先幅度与一致性门槛 + pin)在同一 run 中:top-1 88.5%,工作集覆盖 96.9%,19 次切换(9 次真实切换)。

关于 "平滑":`hybrid_window_topic` 里领域分数的 EMA(`ema_domain_scores`)只写进日志,排序与切换判断读的是未平滑的分数(`DomainScore.confidence = raw_final`);`--autoterm-ema-alpha 0.80` 只作用于语音 query embedding,而论文配置里语音 centroid 权重为 0。所以这套配置下 EMA 不影响路由,压住抖动的是领先幅度门槛(归一化后领先第二名 0.15、领先当前 slice 0.10)、连续 3 次一致、冷却和 pin。taurus_runs.md 里列的 "EMA 0.80" 按此理解。

pin 的反事实(`eval/streaming_sst/evidence/pin_counterfactual.py` → `pin_counterfactual.json`):同一组 w=4 raw 分数上,每个窗口取线上 run 记录的已确认 slice(`active_preset`),比较不 pin(raw 前 4)与 pin(已确认 slice + 其余 raw 前 3)。工作集含正确 slice 的比例:不 pin 85.0%,pin 97.0%,线上实测 96.9%(反事实与线上一致)。router 已确认正确 slice、但它不在 raw 前 4 的窗口有 1,068 个,占已确认正确窗口的 13.8%;这些窗口分成 350 段,中位 2 窗(约 4 秒),60% 的段不超过 2 窗,最长 31 窗。只有 0.2% 落在讲座边界后 60 秒内(全部窗口中这一时段占 3.2%;边界后约 30 秒内已确认的仍是旧 slice,按构造不会计入),所以 pin 挡住的主要是讲座中途的短暂下滑,而不是话题切换时的混乱。局限:open-loop,raw 分数由保存的译文重算,线上 run 的分数缓存刷新节奏未复现。

prototype 文本变体(w=4 前 4 覆盖):原版 85.0、只英文 85.4、只中文 81.0、只 description 83.3、独立改写 89.4、只写领域名 84.0。注意各变体同时改变了语言、条数与语义范围,这是联合敏感性,不是受控的措辞鲁棒性检验。

从 glossary 自动生成 prototype(`scripts/term_memory/generate_router_prototypes.py` → `generated_prototypes.json`;replay 加 `--generated` → `router_replay_generated.json`,同一次运行里原有 24 项结果逐位复现)。现用文本是让 AI 写好后写死在代码里的,当时给 AI 的输入没有留记录:description 一行来自 a1e6033(2026-06-20 PT)的 slice 表;中英两句来自 fe313bd(2026-07-10 PT),同一 commit 还加了 RealSI 领域映射和 load balancing、database 等关键词,NLP 那句也写了 servers、databases、load balancing,所以不只是按领域名写的。这里把每个 1k slice 的术语(`英文 => 中文`,全部给,不抽样)交给 `gpt-5.5-2026-04-23`(带日期的固定快照),生成同结构的三行:description、列 7–10 个核心主题的英文句、它的中文翻译。生成前去掉评测集来源的条目(`acl_tagged_gt` 238 条、`medicine_hard_llm_judge_manual` 212 条),prototype 不接触测试术语;NLP 用剩下的 762 条、medicine 788 条,其余 8 个 slice 都是 Wikipedia/Wikidata 来源,用满 1,000 条。每个 slice 独立生成 10 次,衡量单次措辞的偶然性。

| 指标 | 现用文本 | 自动生成 10 次:中位数 | 范围 | 不低于现用 |
|---|---|---|---|---|
| w=4 前 4 覆盖 | 85.0 | 85.8 | 84.3–91.4 | 8/10 |
| w=4 top-1 | 72.6 | 70.8 | 67.3–77.8 | 3/10 |
| w=8 前 4 覆盖 | 94.8 | 95.0 | 94.2–97.1 | 7/10 |

读法:自动生成与现用文本打平(前 4 覆盖中位 +0.8 点、top-1 中位 −1.8 点),仍落在上面措辞变体的区间里;不同次生成之间相差 7 点,比生成与现用之差大。最先试跑过一次 3 次采样(前 4 覆盖 86.8 / 95.2 / 86.0),样本间差异太大,改为 10 次;试跑的文本已被 10 次结果覆盖,没有保留。生成文本也暴露了 slice 的实际内容:`legal` 被概括为政治、法律与国际事务,`science` 被概括为泛学科分类。局限同上:open-loop、裸相似度、gold 只有 NLP 与 medicine;线上 router 换用生成文本需要闭环重跑才能报告端到端结果。服务已能用 `--autoterm-context-prototypes` 读入生成文本(health 的 `context_similarity.prototypes` 会列出实际载入的文本);闭环用事先定下的第一组生成结果 `configs/router_prototypes_generated_s0.json`(前 4 覆盖 87.0、top-1 68.6),不挑最好的一组,对照是同一硬件上的 AutoTerm 重跑(TERM_ACC 87.30)。运行状态见 `taurus_runs.md`。

子领域测试(NLP、medicine 各拆 8 个子领域,共 24 个 prototype,w=4):同领域 slice 占前 4 槽位 78.3%(随机 33.3%);正确领域进入前 4 的 90.1% 只略高于随机的 82.9%,不能作为证据。肿瘤学讲座中 top-1 为肿瘤学或放疗的无条件比例 48.2%。只有 prototype,没有真实子领域索引,也没有对照专科的 gold,不能回答 "相近领域" 的质疑。

快速切换(NLP / medicine 每 L 秒交替,40 段,裸相似度,w=4):

| L | 前 4 覆盖 | top-1 覆盖 | 新话题进前 4 中位时间 | 进 top-1 中位时间 |
|---|---|---|---|---|
| 15 s | 81.7% | 56.7% | 1.9 s | 5.8 s |
| 60 s | 86.8% | 74.7% | 1.9 s | 5.8 s |
| 300 s | 87.1% | 72.1% | 1.9 s | 5.8 s |

open-loop 不含 "译文 → 路由 → 检索 → 译文" 的反馈,不能替代闭环实验。

## 4. 缺口窗口的术语(`gap_terms.py` → `gap_terms.json`)

2,047 个术语出现中 70 个落在 "正确 slice 不在工作集" 的窗口(NLP 57、medicine 13);命中数 AutoTerm 53、Known 58、Merged-1M 51。这是描述性统计,不是路由造成的因果代价。

## 仍缺的实验(按优先级)

1. 同目录 catalog 的不路由基线:Merged-10k(恰好是这 10 个 1k slice 的并集)、开发集选出的静态 4-slice、随机 4-slice;同样的检索与 prompt 预算。只有两个领域、四个槽位时,静态 NLP+medicine 工作集的覆盖率按构造就是 100%。仓库里 4-domain 10k vs merged-40k 的 pilot 没有显示差异(`multidomain_merged_glossary_pilot_20260710.md`)。
2. 全流水线并发:开着路由与检索的 1 路 / 32 路,统一时钟测首音频到首个有效译文、每 chunk 端到端 p50/p95/p99、积压与超时,并核对路由与检索确实执行。
3. 闭环快速切换且有驱逐压力(相关 slice 多于 4 个),报告边界后 5/10/20 秒的术语准确率、最长缺席时长、不恢复的切换。
4. 线上 router 的逐窗口复现,再做 平滑 × 门槛 × pin 的 2^3 CPU 因子消融。
5. InfiniSST 在同一参考、重分段、术语分母、遮罩与聚合下的重打分(CPU)。
6. ESO 参考译文的双语人工抽检(100–200 段)。
7. merged 基线的阈值与 prompt 准入在独立开发集上调过之后再比较。
8. 冻结配置后在新 talk 上评测;一个新语言对。

## 4. 下一步方向的 CPU 原型(2026-10-26 PT)

两者都是 open-loop:用 7 月审计 run 保存的译文重算 BGE-M3 上下文(w=4,与线上相同编码),不重新解码。嵌入缓存与原始输出在 Taurus `/home/jiaxuanluo/autoterm_taurus_20261024_io/analysis/`。

### 4.1 HMM 话题追踪 router(`hmm_router_replay.py` → `hmm_router_replay.json`)

隐藏状态是当前话题;发射 p(obs_t | d) ∝ exp(s(d|c_t)/T),粘性转移(留 1−ρ,换到其余每个 ρ/(K−1));上下文文本没变的窗口只走转移。工作集选法:`map_pin`(后验最大 + raw 前 3,对应现在的 pin)、`post_top4`(后验前 4)、`mass`(后验累计到 c 的最少 slice,最多 4)。ρ 扫 1e-5–0.1,T 扫 0.005–0.32,c 扫 0.9/0.95/0.99。

真实流上的对照:线上手调规则(闭环实测)覆盖 96.9%、top-1 88.5%、19 次切换、9 次真实切换的延迟中位约 70 秒(27–249 秒);raw 前 4 覆盖 85.0%。

`post_top4` 覆盖率(%,行 T,列 ρ = 1e-5 … 0.1):

| T | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 0.1 |
|---|---|---|---|---|---|
| 0.04 | 97.0 | 96.8 | 96.2 | 95.4 | 93.7 |
| 0.08 | 98.0 | 97.8 | 97.6 | 97.1 | 95.6 |
| 0.16 | 98.4 | 98.3 | 98.2 | 97.9 | 96.9 |
| 0.32 | 98.8 | 98.6 | 98.5 | 98.4 | 97.4 |

同一网格上后验最大领域的切换次数与切换延迟中位:T=0.16、ρ=1e-3 为 34 次 / 29 秒;T=0.16、ρ=1e-2 为 76 次 / 27 秒;T=0.32、ρ=1e-5 为 17 次 / 96 秒。T 越大越稳、越慢,ρ 越大越快、越抖,两个参数给出一条清楚的延迟–稳定性折线;覆盖率在 T ≥ 0.08 的一大片区域都高于线上规则。T=0.005–0.04 时 HMM 明显抖(110–1,227 次切换),说明相邻窗口共享 3/4 上下文、证据不独立,需要较高温度。

自适应工作集(`mass`):平均 1.07 个 slice 覆盖 92.2%,1.44 个 96.4%,1.74 个 96.9%(与线上开满 4 个持平),1.95 个 97.8%。

快切流(rapid_switch_replay 的构造):15 秒一换时粘性设置在 40 段里有 20 段始终没切过去;60–300 秒一换时 `map_pin` 覆盖 91–94%。单一 ρ 不能同时适合讲座级与 15 秒级的切换,这是要报告的取舍。

局限:open-loop;线上规则的数字是闭环、墙钟计时下测的,延迟口径不同(线上取首次进入正确 slice,HMM 取后验最大首次正确);超参在同一组 10 个 talk 上扫,需要留出数据确认。

### 4.2 按术语 embedding 聚类的 slice(`induced_slices_replay.py` → `induced_slices_replay.json`)

目录:十领域 10 万条(89,218 个唯一映射);聚类与中心向量不含评测来源术语(450 条)。每个窗口按 cos(上下文, slice 中心) 依次放入 slice,直到术语预算 B;指标是 2,047 次 gold 术语出现时该术语在工作集里的比例。oracle 按 "该 slice 含多少当前领域的 gold 出现" 排序,给出切法本身的上界。

| 切法 | slice 数 / 大小中位 | B=5k 路由 / oracle | B=10k | B=20k | B=40k |
|---|---|---|---|---|---|
| 分类 + 三行原型 | 10 / 10,000 | 放不下 | 78.3 / 100 | 86.9 / 100 | 92.5 / 100 |
| 分类 + 术语中心原型 | 10 / 10,000 | 放不下 | 78.9 / 100 | 91.1 / 100 | 96.7 / 100 |
| k-means 100 | 100 / 855 | 28.7 / 58.0 | 40.8 / 82.8 | 54.6 / 98.8 | 73.2 / 100 |
| k-means 200 | 200 / 420 | 40.2 / 76.4 | 58.1 / 95.0 | 70.0 / 100 | 81.5 / 100 |
| 随机(同大小分布,200) | 200 / 420 | 6.8 / 37.3 | 13.1 / 53.9 | 24.2 / 74.4 | 46.7 / 93.5 |

读法:

1. 分类 slice 的 oracle 恒为 100%,因为 NLP 与医学两个分类 slice 本来就以评测术语为种子建成,这对分类切法有利。
2. 按术语字符串聚类得到的切分本身不差:k-means 200 只要 11 个簇(8,230 条)就装下 NLP 90% 的 gold 出现、医学 15 个簇(8,996 条),随机切分要 65 个簇(3.4 万条);B=10k 时 oracle 95%。差距在路由:同预算下实际只选中 58%。上下文译文与术语字符串的中心向量不在同一个空间里,簇原型需要换成能和译文上下文对齐的文本(例如每个簇的 LLM 概括)或改用 HMM 累积证据。
3. 分类 slice 用术语中心向量当原型(不用 LLM 写的三行,且不含评测术语)覆盖率持平或更高,原型可以直接从 glossary 得到。

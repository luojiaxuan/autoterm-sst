# 原则化路由:设计与 reviewer 视角自查(2026-10-24 PT 起)

## 要解决的问题

线上 router(`budgeted_top_slices` + `hybrid_window_topic`)由一组手写守卫拼成:EMA 0.80、5 s 更新间隔、6 s warmup、30 s cooldown、一致性窗口、pin。这些计时用的是墙钟 `time.perf_counter()`,所以同一段音频在不同负载下路由结果不同;BGE-M3 加载失败时服务静默继续;评测流只有 NLP + medicine 两个领域,静态 2-slice 并集按构造就有 100% 覆盖。EMNLP demo 审稿意见集中在:显著性、相近领域、未知领域回退、更多语言对。

本方向的目标是把路由换成一个能写进 Method 节、每个部件都能回答"为什么是它"的模型,并在能区分好坏的评测上检验它。

## 模型

### 状态与转移(音频时间)

隐变量 z_t ∈ {1..K} ∪ {u},K 个 catalog 领域加一个显式的 "unknown" 状态 u。两次观测之间相隔 Δt 秒**音频时间**(用 `cursor_s`,不用墙钟),领域以概率 exp(−λΔt) 保持,否则按先验 π 重抽,即连续时间 Markov 链。λ 的含义是 1 / 期望段长,一个参数替代 EMA、warmup、cooldown、一致性窗口四个部件。

### 观测似然

路由上下文(最近 w 个译文 chunk)的 BGE-M3 嵌入 e_t 与每个 slice prototype μ_k 算余弦。

- log p(e | k) = κ · cos(e, μ_k):von Mises–Fisher 似然,省去共享常数;
- log p(e | u) = κ · c_u:unknown 状态的参照相似度。它的含义是"没有任何 prototype 比 c_u 更近时,u 胜出",也就是未知领域回退,不是额外的 if-else;
- 相邻上下文重叠 w − 1 个 chunk,观测不独立。每个观测的对数似然乘 1/w(composite likelihood 的标准修正),每个 chunk 只被计一次。

### 选择 slice

Bayes 决策:装载 slice k 的收益是"它是当前领域"的后验 p_k,成本是常数 τ(占内存、给 prompt 引入噪声),所以 p_k ≥ τ 就装载,超过预算 B 时取后验最高的 B 个。τ 直接就是"一个 slice 至少要以多大概率有用才值得占位",取代固定 top-4。u 的后验质量不装载任何领域 slice;这时系统退化为不带术语表的翻译(或将来的通用回退表)。

### 学习版(对照)

把似然换成判别式:在**不来自评测 talk**的领域标注文本(slice glossary 词条、领域 Wikipedia 句、ESO 训练集讲稿)上训练多项 logistic 回归 p(k | e),外加一个由通用文本构成的 u 类;作为似然使用时除以训练类先验(hybrid HMM 的标准做法)。它回答"手写 prototype 描述是不是瓶颈"。

## 超参数与 sweep 计划

| 超参 | 含义 | 处理 |
|---|---|---|
| κ | 似然温度 | 学习版用 MLE 拟合;原型版做 sweep |
| λ | 切换率(1/期望段长) | sweep 1/30 … 1/3000 s⁻¹,报折线 |
| c_u | unknown 参照相似度 | sweep;用留出领域实验(删掉 medicine prototype)选 |
| τ | slice 成本 | sweep,报覆盖率–平均 slice 数 Pareto 曲线 |
| w | 上下文 chunk 数 | sweep 1/2/4/8 |
| B | 内存预算 | 固定 4(系统约束,不是建模选择) |

所有报告数字用留一对 talk 交叉验证选超参(在 4 对 talk 上选,在第 5 对上报),不在测试 talk 上调。

## 评测设计

1. **粗领域 replay**(已有 run,CPU):与线上 router(覆盖 96.9%,19 次切换)、静态 NLP+med 并集比较覆盖率、平均 slice 数、切换次数。预期:这个评测本身区分不出好坏(静态并集 100%),只用来确认没有退化。
2. **相近领域**:24 个子领域 prototype(NLP 8 + medicine 8 + 其余 8 粗领域)。talk 级子领域标注:ACL dev 五篇 110 对话/语义解析、117 问答、268 词汇借用语料、367 形态感知语言模型、590 高效 Transformer;ESO 五篇 404 肿瘤外科、606 直肠癌多学科、545006 年轻女性乳腺癌、596001 放疗计划评估/医学物理、605000 晚期乳腺癌。注意 ESO 五篇全是肿瘤学,medicine 子领域实际上是"肿瘤学的子专科"。
3. **未知领域**:从 catalog 删掉 gold 领域的 prototype,看 u 是否接管,以及线上 router 在同样设置下把窗口路由到哪里。
4. **驱逐压力**:合成快速切换流(多于 B 个相关 slice),需要真实子领域 slice 才有意义,列为数据阶段的产出。

## Reviewer 攻击自查

| 攻击 | 回答 / 现状 |
|---|---|
| "HMM 也是一堆超参" | 4 个可解释参数替代约 7 个守卫;每个都有 sweep 折线;κ 可 MLE 拟合 |
| "unknown 只是换了名字的阈值" | 是一个阈值,但它有似然解释(参照分布),且只此一个;用留出领域实验选,不在测试流上调 |
| "两个领域的流不能说明路由有用" | 同意。粗领域 replay 只作非退化检查;主证据放在相近领域与驱逐压力评测 |
| "open-loop 不含反馈" | 同意。闭环只在 replay 显示真实差距之后才跑 |
| "prototype 文本是手写的" | 学习版把它换成从标注数据学到的分类器;prototype 文本变体敏感性已有(85.0–89.4%) |
| "子领域标注只有 talk 级、10 个 talk" | 是局限。数据阶段补 ESO 训练集(990 讲,有 SRT)和公开讲座做子领域测试集 |
| "ESO 参考是 LLM 生成的" | zh(Gemini)与 ja(GPT via OpenRouter)是;de 来自数据集自带 `manual_translations/de`,见 `data_provenance.md` |

# 外审与易读性检查的取舍（2026-11-20 PT）

两份输入：GPT-6 Extra High 的事实审计（`gpt6_extra_high.md`），以及无记忆 Opus 的易读性检查（`../2026-11-20-readability/opus_xhigh.md`）。改动在论文 `paper/naacl2027-demo` 与实验分支 `a2553d0`。

## 照改的事实问题

| 来源 | 问题 | 改法 |
| --- | --- | --- |
| GPT 3 | 表 3 的数字在 `summary.json` 里找不到（那里只有中位数） | 新增 `latency_means.py`，从 Taurus 原始记录重算 90 块均值，写进 `latency/single_session_means.json`，与表 3 逐格一致；表注写明是均值 |
| GPT 2.8 | “sentences without terms”不准确：子集按参考译文里没有术语表目标串定义 | 改为“the 935 sentences whose references contain no glossary term” |
| GPT 1.3、易读性 | InfiniSST 的 All 是两语料的不加权平均，其余行是合并计数 | 表注写明 “unweighted means of the two corpora” |
| GPT 4.3 | “removes the preference for early or late positions” 说过了 | 改为正倒序让每个 slice 处在对称位置、抵消位置偏好 |
| GPT 4.4 | “counts each chunk's evidence once” 说过了 | 改为按上下文中新块所占比例给本次更新加权 |
| GPT 2.6、易读性 | 30 段 RealSI 是 10 段讲座的三种顺序 | 正文和表 2 写成 “30 talk instances”，表注写明延迟中位数只在升顶的讲座上算 |
| 易读性 | “nine times as many irrelevant pairs” 的 9 倍是全部注入词对之比 | 改为 “injects nine times as many pairs per chunk” |
| 易读性 | 图 1 的路由框写着 “Sliding-window context similarity”，那是基线 | SVG 改为 “LLM router + sticky topic filter”，重生成 PDF 与 PNG |
| 易读性 | 两个评测流都被叫作“ten-talk stream” | 命名 ACL-ESO stream，全文统一 |
| 易读性 | 相似度路由只在附录定义 | Baselines 加一句：BGE-M3 相似度，连续三次领先且不在 30 秒冷却内才换 |
| 易读性 | 附录 C 说闭环运行先打分再进下一块，与第 4 节的非阻塞更新看似矛盾 | 写明第 6 节闭环用的是逐块先打分，部署时的后台 worker 在输入快于实时时晚一块，回放损失至多 0.33 点 |

另外采纳的措辞：LLM 缩写不再绑定语音模型；摘要与贡献里的 “correct glossary” 改为 “correct topic slice”；引言第三段注明 15.6 秒是我们的路由在 RealSI 上的结果并引用 RealSI；第 3 节删掉此后不再出现的 MaxSim 符号；伦理声明拆开 CC BY 4.0 的指代；附录各段按评语收紧。

## 没跟的

| 来源 | 意见 | 理由 |
| --- | --- | --- |
| GPT 3 | 表 2 的 68.1 应四舍五入为 68.2 | 它读的是两位小数的 68.15；未取整的中位数是 68.149 秒，68.1 正确。结果文件改存三位小数 |
| GPT 1.1 | Holm 校正后 Merged-1M 的 TERM_ACC p=0.0625 | 论文写明以讲座级精确置换检验为准，2026-11-19 已决定不做多重比较校正；BLEU 与 MT-BLEU 校正后仍显著 |
| GPT 1.2 | “on par” 改成 “no statistically detected difference” | 姿态性建议，2026-11-19 已决定保留；区间写在正文 |
| GPT 2.1、2.2、5.1 | 论文应披露 Merged-100k/1M 来自 7 月的 run 1、1M 原表已丢失 | luojiaxuan 决定正文不提 run（`docs/decisions.md`）；同一配置跨 run 的 BLEU 差 0.04 到 0.30，远小于 2.31 与 1.38 的差距；run 1 的输出与打分记录仍在 Taurus，来源写在 `latex/README.md` |
| GPT 2.3、5.3 | 最终版后台 worker 没有并发时延 | 论文不报并发；单会话的检索时间已包含同步打分，后台版只会更快 |
| GPT 2.4、5.3 | LLM 路由在 ACL-ESO 流上的 top-1 是 81.5%，相似度 95.8%，界面上的当前 slice 切换 231 次 | 论文没有关于 ACL-ESO 流 top-1 的说法；界面同时显示后验概率。界面是否平滑显示交给 luojiaxuan 决定 |
| GPT 2.5、2.7 | RealSI 术语来自评测讲座、RealSI 的 TERM_ACC 差 p=0.50 | 论文只用 RealSI 测路由，2026-11-20 已决定不报 RealSI 的 TERM_ACC |
| GPT 4.5 | 未知领域回退失败 | 负结果，不在论文范围；论文只声称从离线目录里选 |
| GPT 4.1、4.2、4.6 | 规模效应的因果、Merged-10k 持平、TERM_ACC 是词面指标 | 正文已写 10k 以内持平；TERM_ACC 沿用 RASST 的定义 |
| 易读性 | 加 Limitations 段 | 研究写作规则：会议不强制时不写 |
| 易读性 | 图 1 加一条“译文回到记忆”的反馈箭头 | 属于图的设计改动，等 luojiaxuan 决定 |

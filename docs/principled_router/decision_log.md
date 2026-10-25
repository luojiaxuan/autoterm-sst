# 决策日志(原则化路由方向)

格式:问题 / 默认选择 / 理由 / 回滚方式 / 外审。

## D1 2026-10-24 PT:replay 的输入用哪个 run
- 选择:7 月审计的 AutoTerm run(`autoterm.json`,SHA 与 evidence README 一致),只用其 `text` 与 `cursor_s`。
- 理由:与线上 router 的 96.9% / 19 次切换出自同一 run,可直接对比。
- 回滚:`embed_replay.py --run` 换任意 run JSON 重算。
- 外审:未进行(见 D4)。

## D2 超参选择协议
- 选择:960 组网格,留一对 talk 交叉验证,目标 = 覆盖率 − 0.05 × slice 数。
- 理由:避免在测试 talk 上调参;0.05 的含义是"一个 slice 至少要换来 5 点覆盖"。系数本身是一个选择,单参数 sweep 已报完整曲线,读者可以按自己的权衡取点。
- 回滚:改 `replay_eval.py` 里的 `lam` 重跑(2.5 分钟,CPU)。

## D3 不上 GPU 闭环
- 选择:本轮不跑闭环。
- 理由:replay 在粗领域上与线上覆盖持平(差距不是"线上缺了什么");唯一可检验的闭环收益是 slice 数减少 3 倍带来的检索降噪,而 Known-domain-1k(1 个 slice)与 AutoTerm 已无检测到的差异,预期收益小于 talk 级检验的分辨率。按规则"replay 显示真实差距之后才跑闭环"。
- 回滚:闭环只需把 `hmm_router.py` 接进 `serve_glossary_capacity_sweep.py` 的 router 接口,用现有 Taurus 启动脚本跑一个会话。

## D4 GPT 外审未能进行
- 原因:Claude in Chrome 已连接,但 chatgpt.com 处于未登录状态,无法选择最高推理档;登录需要用户本人操作。
- 处置:按原判断继续(本轮只有 CPU replay,没有烧算力的决定)。下次可用时补审:design.md、results_replay_20261024.md、D3 与 Findings 判断。

## D5 子领域标注
- 选择:talk 级单标签,ACL 按 Anthology 标题,ESO 按讲座主题(见 data_provenance.md)。
- 理由:零成本、可复核;已在结果里注明粒度局限。
- 回滚:`stress_eval.py` 的 `ITEM_LABEL`。

## D6 2026-10-24 PT:学习版与按 slice 校准是否替换主方法
- 选择:不替换。主方法保留裸余弦 HMM;学习版与校准作为负结果 / 分析写进文档。
- 理由:见 `results_calibrated_20261024.md`。全目录下 logreg 只高 0.4 点覆盖却多 0.2 个 slice;未知领域前沿没有整体前移;伪上下文与真实译文存在 2σ 以上的位置偏移。
- 回滚:`calibrated_eval.py` 已保留三种打分,改 HMM 的 emission 即可换用。
- 外审:仍未进行(chatgpt.com 未登录),补审清单加上本条。

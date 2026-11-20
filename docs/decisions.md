# 决策日志

每条一行：问题、选了什么、为什么、怎么撤回。时间为美国太平洋时间（PT）。

- 2026-11-19 PT｜投稿去向｜按 luojiaxuan 的决定，AutoTerm-SST 改投 NAACL 2027 System Demonstrations（12-04 AoE 截稿），不扩展成 RASST 后续的 main paper｜他的一作重心转向多模态与 RL，demo 收尾成本低；按效用选术语、只注入模型会译错的词等扩展方向需要数月和新的评测设定｜撤回：在 `paper/naacl2027-demo` 之外另开分支写 main paper，demo 稿不受影响。
- 2026-11-19 PT｜论文源码｜以 5afa7ab（即 arXiv v1 与 EMNLP 投稿版）为底，恢复到 `latex/`，样式文件换成 acl-org/acl-style-files `d5adc82` 的官方版｜main 分支只留 PDF，重投需要源码｜撤回：删掉 `latex/` 即可。
- 2026-11-19 PT｜32 路并发的说法｜从摘要和贡献列表删掉“vLLM host sustains 32 real-time sessions”，正文也不放 10-24 的全流水线并发（G1）结果｜luojiaxuan 认为服务端并发与本文贡献无关；7 月那次只测了生成，打开路由与检索后两张 A6000 只能撑 8 路实时，留着原句会被读成整套系统的能力｜撤回：恢复 5afa7ab 的摘要句，并在正文补上 G1 的 8 路数字。
- 2026-11-19 PT｜单会话延迟｜正文加入单会话的首个译文时间（2.3 秒）和每块端到端延迟（p50 0.57 秒、p95 1.17 秒），取自 G1 的 N=1、vllm serve、BGE-M3 在 GPU 的那一档｜审稿人 FZMM 明确要用户可感知的延迟，这是单会话指标，与并发无关｜撤回：删掉 Latency and speed 段的第二句。
- 2026-11-19 PT｜小规模合并基线｜正文和表 1 加入第二次运行的 Merged-2k、Merged-10k；写法是“合并术语表在 10k 以内与 AutoTerm-SST 持平，再大就落后”，Merged-10k 名义上高 0.8 TERM_ACC 点（p=0.08）照实写｜同硬件对照是审稿人会问的“为什么不直接合并”，与引言“合并在中等规模可用”一致｜撤回：删掉表 1 的 Run 2 后两行和 Effect of catalog scale 段的第二句。
- 2026-11-19 PT｜Limitations 段｜删掉｜NAACL 2027 demo 的 CFP 不要求，且计入 6 页正文；“评测术语在 slice 里”这一范围说明已在 Data 段正面写了一次｜撤回：从 5afa7ab 恢复该段。
- 2026-11-19 PT｜ESO 中文参考的来源｜正文写“RASST 的模型生成参考”，不写具体模型名｜RASST 论文写的是 GPT-5.4，而 `docs/principled_router/data_provenance.md`（在 explore/principled-router 分支）核对处理脚本后记的是 Gemini，两处冲突，需要问 jiaxingxu｜撤回：确认后写回具体模型名。
- 2026-11-19 PT｜界面截图｜图 2 换成 `scripts/render_ui_replay.py` 的 3 倍渲染：真实网页界面回放 run 1（7 月审计运行）事件 600–639 的记录，图注写明是回放｜旧截图来自 10k slice 配置、带鼠标和提示条；Taurus 卡被占，现场跑一次 GPU 会话代价大，回放的内容全是已评测运行的真实输出｜撤回：用 `git show 5afa7ab:demo_paper_emnlp/latex/figures/ui_evidence_panel.png` 换回旧图，或在 4K 屏上现场截图后替换。
- 2026-11-19 PT｜失败案例句｜删掉 “ACL 到 medicine 边界后 medicine slice 晚进工作集” 一句｜写作 skill 规定失败案例不进投稿版，且正文需要压回 6 页｜撤回：从 6bddc1e 的 Routing behavior 段恢复。
- 2026-11-19 PT｜GPT 外审档位｜用 GPT-6 的 Extra High（Pro 下一档）｜luojiaxuan 说 demo short paper 不用 Pro｜撤回：重新用 Pro 档审一次。外审意见照改的：Table 4 生成 prototype 的 Top-1 中位数 70.8→70.7；“比 InfiniSST 高约 27 点”跨计分口径，改为定性；203 ms 注明包含语音编码；生成 prototype 改为 “comparable routing accuracy”；界面只说 “能看到” 错误来源；附录 A 写明显著性以置换检验为准、说明 Merged-10k 的 bootstrap 区间不跨零而 p=0.084；附录 B 写明 run 2 中 AutoTerm-SST 单独占一台服务。没跟的：把 on par 改成 “no statistically detected difference”、Holm 校正、恢复 Limitations 段、在正文披露更多不利细节（判据：同行读原说法不会被误导，相应事实已在附录表格里）。

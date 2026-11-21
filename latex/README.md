# AutoTerm-SST 论文（NAACL 2027 System Demonstrations）

投稿截止 2026-12-04 AoE；正文 6 页，附录不超过 2 页，伦理声明与参考文献不限。`acl_latex.tex` 是入口，各节在 `sections/`；`bash latex/make_arxiv.sh <out.tar.gz>` 生成去掉注释、带 `.bbl` 的 arXiv 源码包并验证页数。

## 数字从哪来

正文和表格不区分 run，统一以 AutoTerm-SST（语言模型路由）为对照。各 run 播放同一音频顺序、切块边界相同（timing signature `7c0be6da…` 一致），跨 run 的比较按讲座逐一配对。下表的路径都在实验分支 `explore/llm-router`（`28c03b4` 起）。

| 论文位置 | 来源 run | 结果文件 |
| --- | --- | --- |
| 表 1：AutoTerm-SST、Known-domain-1k、Similarity routing、Merged-2k、Merged-10k | run 3：2026-11-20，hyper01 两张 H200 | `docs/llm_router_20261120/run3/results.json`、`prompt_precision.json` |
| 表 1：Merged-100k、Merged-1M | run 1：2026-07-11，单张 H200；输出在 Taurus `/mnt/data1/jiaxuanluo/autoterm_streamlaal_20260711/` | `docs/autoterm_1m_budget_search_20260711.md`（explore/multidomain-routing） |
| 表 1：InfiniSST | RASST 的 no-RAG 五讲座结果，两个语料取平均 | 同上文档的 InfiniSST 行 |
| 表 1 的 † 与附录统计表：run 3 内的配对 | run 3 | `run3/paired_significance.json`、`run3/term_free_quality.json` |
| 表 1 的 † 与附录统计表：对 Merged-100k、Merged-1M 的配对 | run 3 的 AutoTerm-SST 对 run 1 的输出 | `run3/vs_run1_merged_paired_significance.json`、`run3/vs_run1_merged_term_free_quality.json`；脚本 `scripts/taurus_run3_vs_run1_merged.sh` |
| 表 3 与图 3：RealSI 路由 | 2026-11-20，hyper01 两张 H200，三种顺序 | `docs/realsi_20261120/results/routing_coverage.json`；图由 `eval/streaming_sst/render_routing_states_svg.py` 生成 |
| 表 4：处理速度 | 2026-11-20，Taurus 三张 A6000，单会话一场 ACL 讲座 | `docs/llm_router_20261120/latency/summary.json` 的 1 路一档 |
| 表 2（回放表）、第 6 节 topic filter 段与附录路由诊断 | run 1 存下的译文，离线回放 | `docs/llm_router_20261120/README.md` 各节与同目录 JSON |
| 图 2：界面截图 | 回放 run 3 第 600 到 639 块的输出；状态行的后验由 `router_posterior_replay.py` 在 aries 一张 A6000 上按线上节奏在录下的译文上重放，前 660 块里首位 slice 与线上一致 87%，图中 40 块全部一致 | 实验仓库 `feat/status-row-top-slices`（[PR #2](https://github.com/luojiaxuan/autoterm-sst/pull/2)）的 `scripts/render_ui_replay.py`；后验在 `docs/llm_router_20261120/ui_posterior_replay.json` |

## 没进论文的

run 1、run 2 里相似度路由对 Known-domain-1k、Merged-2k、Merged-10k 的配对（run 1 在 `docs/evidence_20261024/paired_significance.json`，run 2 在同目录 `taurus/paired_significance.json`）只留在实验仓库，论文不报。决定与撤回方式见 `docs/decisions.md`。

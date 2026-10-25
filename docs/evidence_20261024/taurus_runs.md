# Taurus 补充实验:运行记录(2026-10-24 PT 起)

目的:补两类证据。(1) G1:开着路由与检索时 1/8/16/32 路实时并发的客户端可见延迟;(2) 同目录不路由基线:十个 1k slice 的并集(Merged-10k)、只放 NLP+medicine 两个 slice 的并集(Merged-NLP+Med-2k)、按 talk 换领域的 Known-domain-1k,以及同一硬件上重跑 AutoTerm-1k×4 作对照。

## 主机与资源

- 主机:Taurus(8×RTX A6000 48GB)。通过 Slurm 运行:所有进程都是 luojiaxuan 自己的占位作业 48285(`sglang-omni-hold`,6 GPU / 48 CPU / 300G)里的 job step(`srun --jobid=48285 --overlap --gres=gpu:2`)。GPU 6、7 属于 zerucheng 的作业,未使用。
- 工作目录:`/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/`(`/mnt/data1` 已满,`/mnt/data2` 剩约 20G,产物都很小)。代码是 `luojiaxuan/autoterm-sst` 分支 `explore/evidence-upgrade` 的 clone。
- 环境:`~/miniconda3/envs/spaCyEnv`(vLLM 0.13.0)。模型 `speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf`,retriever `rasst-hn1024.pt`(`/mnt/data1/.../autoterm_capacity_100topics_20260710/models/`)。BGE-M3 用本地快照 `5617a9f61b028005a4858fdac845db406aefb181`(缓存的 `main` 指向的快照没有 tokenizer,离线时会加载失败;加载失败时服务会 "continuing without it",因此每次启动都要求 health 里 `context_similarity` 为 `ready`)。

| tmux | Slurm step | GPU | 端口 | 用途 |
|---|---|---|---|---|
| sst-c | 48285.13 | 0,1 | 8033 | G1 并发(routed,max_num_seqs 32,显存占比 0.78) |
| sst-a | 48285.14 | 2,3 | 8031 | AutoTerm-1k×4 闭环重跑(routed,单会话) |
| sst-b | 48285.15 | 4,5 | 8032 | 不路由基线(fixed;Merged-10k、Merged-NLP+Med-2k、Known-domain-1k 三个会话并行) |

AutoTerm 单独占一台服务:router 的 warmup / update / cooldown 计时用的是墙钟(`time.perf_counter()`),和其他会话共享 GPU 会改变路由行为。不路由的三个基线没有这个问题,可以并行。

## Slice 与基线的构造

- 十个 1k slice:`scripts/term_memory/build_nested_prefix_sweep.py` 从 `catalog_10domain_100k_deepcat_20260710` 的十个 10k slice 取前 1,000 行(glossary 与 MaxSim 索引同步切片,不重新编码)。核对:论文 AutoTerm run 检索到的 1,331 个不同术语全部落在这些前缀里,其他 catalog 版本对不上。
- `scripts/term_memory/build_slice_union_baselines.py`:Merged-10k = 十个 1k slice 按序合并,只去掉完全相同的 (规范化 source, 中文 target) 对(去掉 120 条,剩 9,880),保留冲突译法;Merged-NLP+Med-2k = nlp_core_1k + medicine_core_1k(2,000 条)。服务用的 manifest:`slices/manifest_taurus_1k.json`。

## 服务与客户端配置

- 服务:`scripts/taurus_evidence_server.sh`(包装 `eval/streaming_sst/serve_glossary_capacity_sweep.py`)。AutoTerm 参数按论文与 7 月正式 run 的启动脚本重建:`budgeted_top_slices`、term budget 4000、最多 4 个 slice、context 窗口 4 chunk、只用 context similarity(权重 1.0,其余 0)、EMA 0.80、update 5s / warmup 6s / cooldown 30s、candidate budget 100、MaxSim 阈值 0.78、prompt top-10。7 月 run 的完整配置文件没有留存,这是重建值。
- 闭环客户端:`scripts/taurus_evidence_eval.sh`,与 7 月审计 run 相同(alternating、seed 20260707、5 ACL + 5 ESO、1.92s chunk、backpressure 1 chunk、stall 60s、idle 5s)。
- G1 客户端:`eval/streaming_sst/concurrency_full_pipeline.py`。每个会话用不同的真实讲座音频,按实时节奏每 1.92 秒发一个 chunk(不等回复);以 `meta.cursor_samples` 把每条输出对应回完成它的 chunk,记录 "chunk 发出 → 对应输出到达" 的延迟、首段音频到首个译文的时间、最后一个 chunk 之后的排空时间、服务端检索/生成耗时、每条输出是否带路由与检索证据、GPU 显存。各会话起始时间在一个 chunk 内错开。

## 进度信号、告警与收尾

- 进度:服务日志里的会话活动行;闭环客户端结束时写 `runs/<name>.json`;G1 每完成一个并发档写一次 `runs/g1_full_pipeline.json`。
- 停滞阈值:闭环 run 期间服务日志 15 分钟没有新输出,或 tmux 会话消失、日志出现 Traceback / out of memory / Killed,即视为失败。单会话闭环 run 不能断点续跑,失败则整段重跑。
- 告警:本 session 的监视器每 5 分钟检查一次上述信号。
- 收尾:run 结束、结果拉回并核对后,关闭 sst-a/b/c 三个 tmux(随之结束 job step),占位作业 48285 本身保持不动。

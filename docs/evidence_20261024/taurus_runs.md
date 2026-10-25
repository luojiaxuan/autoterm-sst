# Taurus 补充实验:运行记录(2026-10-24 PT 起)

目的:补两类证据。(1) G1:开着路由与检索时 1/8/16/32 路实时并发的客户端可见延迟;(2) 同目录不路由基线:十个 1k slice 的并集(Merged-10k)、只放 NLP+medicine 两个 slice 的并集(Merged-NLP+Med-2k)、按 talk 换领域的 Known-domain-1k,以及同一硬件上重跑 AutoTerm-1k×4 作对照。

## 主机与资源

- 主机:Taurus(8×RTX A6000 48GB)。通过 Slurm 运行:所有进程都是 luojiaxuan 自己的占位作业 48285(`sglang-omni-hold`,6 GPU / 48 CPU / 300G)里的 job step(`srun --jobid=48285 --overlap --gres=gpu:2`)。GPU 6、7 属于 zerucheng 的作业,未使用。
- 工作目录:`/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/`(`/mnt/data1` 已满,`/mnt/data2` 剩约 20G,产物都很小)。代码是 `luojiaxuan/autoterm-sst` 分支 `explore/evidence-upgrade` 的 clone。
- 环境:`~/miniconda3/envs/spaCyEnv`(vLLM 0.13.0)。模型 `speech_llm_zh_cap16_denoise_budget_ttag_r32a32_ep1_taurus4_hf`,retriever `rasst-hn1024.pt`(`/mnt/data1/.../autoterm_capacity_100topics_20260710/models/`)。BGE-M3 用本地快照 `5617a9f61b028005a4858fdac845db406aefb181`(缓存的 `main` 指向的快照没有 tokenizer,离线时会加载失败;加载失败时服务会 "continuing without it",因此每次启动都要求 health 里 `context_similarity` 为 `ready`)。

| tmux | GPU(Slurm 分配) | 端口 | 用途 |
|---|---|---|---|
| sst-c | 2 张 | 8033 | G1 并发(routed,max_num_seqs 32,显存占比 0.78,8 CPU) |
| sst-a | 2 张 | 8031 | AutoTerm-1k×4 闭环重跑(routed,单会话,12 CPU) |
| sst-b | 2 张 | 8032 | 不路由基线(fixed;Merged-10k、Merged-NLP+Med-2k、Known-domain-1k 三个会话并行,12 CPU) |
| sst-ea / sst-eb / sst-g1 | 无 | | 客户端 step(`--gres=none`) |

全部由 `scripts/taurus_evidence_launch_all.sh` 一次拉起;实际 step 号与 GPU 号见各服务日志首行。

启动时踩到的三件事:
1. `srun --overlap` 只让 step 共享 CPU。服务 step 不写 `--gres` 时会拿走分配里全部 6 张卡,后面的 step 永远排队;现在每个服务 step 显式 `--gres=gpu:2`,客户端 `--gres=none`。
2. 这个分配里各 step 的 CPU 加起来不能超过 48,超过的 step 静默排队;服务 CPU 已压到 8/12/12。
3. 三个数据盘全满(`/mnt/data` 3.8G、`/mnt/data1` 232K、`/mnt/data2` 2.9G 且在被别的租户持续写入),只有 `/` 有 336G。为避免结束时写结果失败,日志、run JSON 与服务 tmp 放在 `/home/jiaxuanluo/autoterm_taurus_20261024_io/`(违反 "不落家目录" 的约定,作为临时例外记录在案;结果拉回并核对后删除)。代码与 slice 仍在 `/mnt/data2`。

AutoTerm 单独占一台服务:router 的 warmup / update / cooldown 计时用的是墙钟(`time.perf_counter()`),和其他会话共享 GPU 会改变路由行为。不路由的三个基线没有这个问题,可以并行。

## Slice 与基线的构造

- 十个 1k slice:`scripts/term_memory/build_nested_prefix_sweep.py` 从 `catalog_10domain_100k_deepcat_20260710` 的十个 10k slice 取前 1,000 行(glossary 与 MaxSim 索引同步切片,不重新编码)。核对:论文 AutoTerm run 检索到的 1,331 个不同术语全部落在这些前缀里,其他 catalog 版本对不上。
- `scripts/term_memory/build_slice_union_baselines.py`:Merged-10k = 十个 1k slice 按序合并,只去掉完全相同的 (规范化 source, 中文 target) 对(去掉 120 条,剩 9,880),保留冲突译法;Merged-NLP+Med-2k = nlp_core_1k + medicine_core_1k(2,000 条)。服务用的 manifest:`slices/manifest_taurus_1k.json`。

## 服务与客户端配置

- 服务:`scripts/taurus_evidence_server.sh`(包装 `eval/streaming_sst/serve_glossary_capacity_sweep.py`)。AutoTerm 参数按论文与 7 月正式 run 的启动脚本重建:`budgeted_top_slices`、term budget 4000、最多 4 个 slice、context 窗口 4 chunk、只用 context similarity(权重 1.0,其余 0)、EMA 0.80、update 5s / warmup 6s / cooldown 30s、candidate budget 100、MaxSim 阈值 0.78、prompt top-10。7 月 run 的完整配置文件没有留存,这是重建值。
- 闭环客户端:`scripts/taurus_evidence_eval.sh`,与 7 月审计 run 相同(alternating、seed 20260707、5 ACL + 5 ESO、1.92s chunk、backpressure 1 chunk、stall 60s、idle 5s)。
- G1 客户端:`eval/streaming_sst/concurrency_full_pipeline.py`。每个会话用不同的真实讲座音频,按实时节奏每 1.92 秒发一个 chunk(不等回复);以 `meta.cursor_samples` 把每条输出对应回完成它的 chunk,记录 "chunk 发出 → 对应输出到达" 的延迟、首段音频到首个译文的时间、最后一个 chunk 之后的排空时间、服务端检索/生成耗时、每条输出是否带路由与检索证据、GPU 显存。各会话起始时间在一个 chunk 内错开。

## 十 talk 闭环结果(同一硬件,2026-10-25 UTC)

四个 run 都在 Taurus 同一批 A6000 上、同一份代码、同一条十 talk 交替流上跑,评分管线与 7 月审计 run 完全一致(先用 7 月 AutoTerm run 复核:BLEU 45.090898、MT-BLEU 42.2827、分句文件逐字节相同)。TERM_ACC 是 2,047 个术语出现的命中率,prompt precision 用论文的 raw 口径。

| 设置 | 需要会话前输入 | TERM_ACC | NLP | Medicine | BLEU | MT-BLEU | Prompt precision | Refs/chunk |
|---|---|---|---|---|---|---|---|---|
| AutoTerm-1k×4(路由) | 无 | 87.30 | 87.94 | 86.01 | 45.15 | 42.36 | 35.2% | 0.94 |
| Merged-10k(十个 slice 合并,不路由) | 无 | 88.13 | 89.11 | 86.16 | 44.97 | 42.19 | 27.1% | 1.26 |
| Merged-NLP+Med-2k(两个相关 slice 常开) | 要知道会出现哪两个领域 | 87.69 | 88.89 | 85.27 | 45.18 | 42.42 | 39.2% | 0.86 |
| Known-domain-1k(按 talk 换领域) | 要逐 talk 指定领域 | 87.98 | 88.89 | 86.16 | 44.79 | 41.94 | 48.8% | 0.69 |

配对检验(talk 级置换 p,分层 talk bootstrap 95% CI):

| 对比 | ΔTERM_ACC | ΔBLEU | ΔMT-BLEU |
|---|---|---|---|
| AutoTerm − Known | −0.68 [−1.48, +0.00],p=0.18 | +0.36 [−0.24, +1.26],p=0.69 | +0.42,p=0.66 |
| AutoTerm − Merged-10k | −0.83 [−1.58, −0.25],p=0.084 | +0.18 [−0.18, +0.64],p=0.43 | +0.17,p=0.51 |
| AutoTerm − Merged-2k | −0.39 [−1.29, +0.30],p=0.49 | −0.03,p=0.94 | −0.06,p=0.79 |

无术语句(参考里没有 glossary 串)上的 BLEU 差:AutoTerm − Known +0.94(p=0.22)、− Merged-10k +0.49(p=0.17)、− Merged-2k −0.11(p=0.79),都不显著。

读法:

1. **在这个只有两个领域、十个 1k slice 的设置里,路由没有带来可测的好处**:把十个 slice 直接合并成 10k 不路由,术语准确率名义上还高 0.8 点(talk 级 p=0.084),BLEU 无差异;它往 prompt 里多塞了 34% 的条目、精度从 35% 降到 27%,但 10k 规模下这点噪声没有造成可测的质量损失。论文对 Merged-100k / 1M 的优势(尤其 1M)仍然成立,但 "路由是必要的" 这个结论需要把合并规模从 10k 推到 100k 以上、或者在同一 catalog 扩大时比较,才站得住。
2. AutoTerm 与 Known-domain 在同一硬件上依然没有检测到差异(第二次独立 run 的结论与 7 月一致)。
3. AutoTerm 同配置换硬件重跑:TERM_ACC 88.32 → 87.30,BLEU 45.09 → 45.15,MT-BLEU 42.28 → 42.36,工作集覆盖 96.9% → 98.0%。run 间与硬件间差异约 ±1 TERM_ACC 点,与上面各对比的量级相当。

## G1 结果:开着路由与检索的实时并发(2×A6000)

每档每个会话 180 秒真实讲座音频,按 1.92 秒一块实时送入;"延迟" 是某块音频送出到对应译文到达客户端的时间(前 3 块不计);"超时占比" 是延迟超过 1.92 秒的块所占比例;"排空" 是最后一块送出后还要等多久才收到最后的译文(持续增长说明跟不上)。每档开始前有一次单会话预热,不计入。摘要见 `g1_serving_summary.json`。

| 配置 | N | 延迟 p50 / p95 / p99(秒) | 超时占比 | 首个译文(秒) | 排空(秒) | 检索 p50 | 生成 p50 |
|---|---|---|---|---|---|---|---|
| 进程内 vLLM,BGE-M3 在 CPU | 1 | 0.60 / 1.35 / 1.57 | 0% | 2.3 | 1.6 | 0.06 | 0.41 |
| | 8 | 3.11 / 4.69 / 5.42 | 93% | 2.3 | 2.6 | 0.35 | 1.00 |
| | 16 | 8.51 / 13.8 / 14.7 | 98% | 2.6 | 10.3 | 0.88 | 2.40 |
| | 32 | 15.9 / 26.6 / 29.3 | 100% | 3.1 | 19.6 | 1.88 | 2.88 |
| vllm serve,BGE-M3 在 CPU(8 CPU) | 1 | 0.54 / 1.65 / 2.46 | 3% | 2.2 | 0.5 | 0.07 | 0.31 |
| | 8 | 5.89 / 13.0 / 15.1 | 88% | 2.3 | 0.9 | 1.31 | 0.91 |
| | 16 | 6.17 / 11.7 / 13.5 | 97% | 2.4 | 7.2 | 1.60 | 1.48 |
| | 32 | 12.5 / 20.1 / 22.5 | 100% | 2.8 | 8.9 | 4.85 | 2.84 |
| vllm serve,BGE-M3 在 GPU(12 CPU) | 1 | 0.57 / 1.17 / 1.55 | 0% | 2.3 | 0.7 | 0.12 | 0.31 |
| | 8 | 0.82 / 1.74 / 2.20 | 3% | 2.4 | 0.7 | 0.09 | 0.43 |
| | 16 | 1.25 / 5.17 / 9.72 | 25% | 2.4 | 1.5 | 0.16 | 0.53 |
| | 32 | 4.27 / 16.9 / 22.7 | 95% | 2.8 | 4.0 | 0.61 | 1.25 |

三档里 retriever 都和 vLLM 的第一张卡共用 GPU。读法:

- 论文里 "32 路实时" 的那次测试只测了生成(外部 vllm serve、不加载 glossary、客户端每 0.45 秒发 0.5 秒音频,只统计服务端生成时间)。打开路由与检索后,同样两张 A6000 在 8 路时仍是实时的(p95 1.74 秒),16 路中位数可以但尾部不行,32 路跟不上。
- 进程内 vLLM 后端在并发下排队严重,并发场景要用 vllm serve。
- 路由用的 BGE-M3 放在 CPU 是单会话下的设置,并发时成为瓶颈(8 路检索中位数 1.31 秒);挪到 GPU 后降到 0.09 秒。
- 首个译文稳定在 2.2–2.8 秒(冷启动那一次 13–14 秒,已作为预热丢弃)。
- 给检索与路由单独一张卡(论文原始的三卡布局)的那一档没有跑成:另一个用户在 Slurm 之外起的 SGLang 服务占了本分配里的 GPU 0–3,只剩两张卡。需要时用 `scripts/taurus_evidence_chain_final.sh` 的后半段在三张空卡上重跑。

## 进度信号、告警与收尾

- 进度:服务日志里每个 batch 一行 `RASST_BATCH_METRIC`;闭环客户端结束时写 `runs/<name>.json`;G1 每完成一个并发档写一次 `runs/g1_full_pipeline.json`(都在 IO 根目录下)。
- 停滞阈值:闭环 run 期间服务日志 15 分钟没有新输出,或 tmux 会话消失、日志出现 Traceback / out of memory / Killed,即视为失败。单会话闭环 run 不能断点续跑,失败则整段重跑。
- 告警:本 session 的监视器每 5 分钟检查一次上述信号。
- 收尾:run 结束、结果拉回并核对后,关闭 sst-a/b/c 三个 tmux(随之结束 job step),占位作业 48285 本身保持不动。

## 收尾状态(2026-10-25 UTC)

- 所有 job step 与进程已结束,占位作业 48285 未改动。
- 结果目录已从家目录挪回 `/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/io/`(runs、scores、bundles、logs,176 MB;挪动前后整体哈希一致,家目录副本已删除)。
- 汇总结果与分析输出已进 Git(本目录与 `taurus/` 子目录);原始 run JSON 含 ESO 讲座派生的中文译文,公开上传前需确认源数据许可,状态 HF pending,正本在上述 Taurus 目录。

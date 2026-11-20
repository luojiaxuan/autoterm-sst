# 语言模型路由的离线回放（2026-11-20）

在 7 月审计的十 talk 运行（run 1，8,776 个解码窗口）存下的中文译文上回放路由，不重新解码。语言模型是 `Qwen/Qwen3-4B-Instruct-2507`（bf16，transformers），在 aries 借用的一张 A6000 上跑。打分是选项字母作为第一个回答 token 的概率，正序与倒序各算一次取平均。余弦路由的数字来自 `docs/evidence_20261024/router_replay.json` 与 `rapid_switch.json`，同一批窗口、同一 w=4。

| 路由（w=4，不加 pin） | top-1 | 前 4 覆盖 | top-1 变化次数 |
| --- | --- | --- | --- |
| 余弦（BGE-M3 原型） | 72.6 | 85.0 | 1,809 |
| 语言模型字母概率 | 79.4 | 90.5 | 986 |

快速切换流（NLP 与 medicine 每 L 秒交替，40 段）：

| L | 余弦 前 4 / top-1 | 语言模型 前 4 / top-1 | 新话题进 top-1 的中位秒数：余弦 / 语言模型 |
| --- | --- | --- | --- |
| 15 s | 81.7 / 56.7 | 90.4 / 64.1 | 5.76 / 3.84 |
| 30 s | 84.9 / 68.8 | 87.8 / 75.2 | 5.76 / 5.76 |
| 60 s | 86.8 / 74.7 | 93.6 / 82.5 | 5.76 / 3.84 |
| 120 s | 85.8 / 71.6 | 92.7 / 80.6 | 5.76 / 5.76 |
| 300 s | 87.1 / 72.1 | 93.4 / 83.0 | 5.76 / 5.76 |

粘性 HMM 滤波（`llm_hmm_filter.py`）：语言模型概率当作观测，指数 1/w 抵消相邻窗口共享的 3/4 上下文，唯一参数是每块的话题切换概率 ρ。

| ρ | top-1 | 前 4 覆盖 | top-1 变化次数 |
| --- | --- | --- | --- |
| 1e-5 | 82.0 | 98.9 | 183 |
| 1e-4 | 81.6 | 98.9 | 231 |
| 1e-3 | 80.9 | 98.7 | 285 |
| 1e-2 | 79.9 | 98.6 | 376 |
| 1e-1 | 78.6 | 98.4 | 460 |

参照：线上手调规则（领先阈值、连续三次、冷却与 pin）在同一 run 闭环实测前 4 覆盖 96.9%、确认 slice 正确 88.5%。

原始输出：`llm_router_replay.json`、`llm_hmm_filter.json`；逐窗口概率 `result.w4.npy` 在 aries `/mnt/data6/jiaxuanluo/omni-eval/runs/20261120T040902Z/`（含 ESO 派生译文的输入只留在私有存储）。

## 快速切换下的 HMM 滤波

`llm_rapid_hmm.py`（aries 借用的一张 A6000，run `20261120T060611Z`）：与上面快速切换流相同的切法，存下逐窗口概率后加 HMM（ρ 同上一网格）。下表是 ρ = 1e-3；新话题进前 4 的中位秒数在两种设置下都是 0 到 1.92 秒，40 段里新话题都进过前 4。

| L | 语言模型原始 前 4 / top-1 | 语言模型 + HMM 前 4 / top-1 | 新话题升到 top-1 的中位秒数：原始 / HMM | HMM 下 top-1 始终没轮到的段数 |
| --- | --- | --- | --- | --- |
| 15 s | 90.4 / 64.1 | 99.7 / 43.9 | 3.84 / 7.68 | 5 |
| 30 s | 87.8 / 75.2 | 97.0 / 61.5 | 5.76 / 9.60 | 2 |
| 60 s | 93.6 / 82.5 | 96.9 / 75.8 | 3.84 / 9.60 | 1 |
| 120 s | 92.7 / 80.6 | 97.0 / 79.1 | 5.76 / 11.52 | 0 |
| 300 s | 93.4 / 83.0 | 98.9 / 83.3 | 5.76 / 9.60 | 0 |

读法：HMM 把前 4 覆盖抬到 97% 以上，代价是 top-1 换得慢 4 到 6 秒。工作集有四个位置，两个话题交替时都留在里面，检索用到的正是前 4。原始输出 `llm_rapid_hmm.json`。

## 德语与日语

RASST 论文的 De、Ja 运行（`release_profiles_taurusdata2_20260528T000922Z/paper_adjacent_cache30`，lm2，已知领域术语表）按发出时间切成 1.92 秒一块，五个 ACL talk 与五个 ESO 讲座交替拼成一条流。路由、原型与提示和中文相同，只把提示里的目标语言换成 German 或 Japanese。

| 目标语言 | 余弦 top-1 / 前 4 / 变化 | 语言模型 top-1 / 前 4 / 变化 | 语言模型 + HMM 前 4 覆盖（ρ = 1e-5 / 1e-3 / 1e-1） |
| --- | --- | --- | --- |
| zh（run 1） | 72.6 / 85.0 / 1,809 | 79.4 / 90.5 / 986 | 98.9 / 98.7 / 98.4 |
| de | 68.2 / 84.8 / 2,192 | 78.0 / 87.7 / 1,076 | 97.8 / 97.5 / 96.8 |
| ja | 69.2 / 84.9 / 1,985 | 79.0 / 88.2 / 1,008 | 98.5 / 98.3 / 97.5 |

原始输出：`multilingual_de.json`、`multilingual_ja.json`、`multilingual_hmm_filter.json`。

## 近领域：语言模型切出的细粒度 slice

`llm_fine_slices.py` 用同一个 4B 模型把 NLP 与医学的 10k 目录各切成 8 个子方向（先从 400 条非评测来源的术语提出子方向，再给每条术语按字母概率归类），加上其余 8 个领域，共 24 个 slice。子方向名称与大小见 `fine_slices_taxonomy.json`。指标是 2,047 次 gold 术语出现中，它所在的 slice 在该块工作集前 4 里的比例（`fine_slice_replay.py`、`fine_slices_coverage.json`）。

| 路由 | 全部 | NLP | 医学 |
| --- | --- | --- | --- |
| 余弦（子方向描述作原型） | 51.6 | 43.1 | 68.6 |
| 语言模型 | 73.9 | 80.6 | 60.2 |
| 语言模型 + HMM（ρ = 1e-5） | 83.9 | 90.3 | 71.0 |
| 每个 talk 取最好的 4 个 slice（上界） | 97.3 | 97.7 | 96.5 |

医学讲座的 gold 术语分散在操作与治疗（211）、药理（211）、疾病（141）、诊断工具（58）、遗传（47）五个子方向，路由要在相邻子方向里挑对组合。

## 三语 multi-LoRA 服务的一致性

`scripts/radix_stream_eval/lora_check.sh` 与 `lora_noise_floor.sh`（b200 访客两张卡，vLLM 0.34.0，每个服务一张卡）：40 句 ACL 英文句子，合并模型的贪心输出作目标，再看三种服务方式离目标多远。“平均差”是对目标每个 token 的对数概率之差的绝对值的平均（单位 nat）。

| 语言 | base + 还原的 adapter：逐字相同 / 平均差 | 同一合并模型换 eager 核：逐字相同 / 平均差 | base 不加 adapter：逐字相同 / 平均差 |
| --- | --- | --- | --- |
| zh | 30 / 0.147 | 31 / 0.145 | 3 / 1.366 |
| de | 30 / 0.140 | 31 / 0.126 | 6 / 1.311 |
| ja | 28 / 0.116 | 28 / 0.132 | 4 / 1.230 |

读法：adapter 服务与合并模型之间的差，和同一合并模型只换计算核（CUDA graph 换成 eager）的差一样大，属于数值噪声；MoE 的专家选择对微小数值差敏感，所以句子级会出现近义改写（不同句对的字符相似度中位数 0.92–0.93）。base 不加 adapter 时差约 1.3 nat，adapter 抹掉了约 89% 的差距，剩下的就是这层噪声。de 用精确 SVD、zh 与 ja 用随机化 SVD，三者落在同一水平。同一 LoRA 服务两次跑 zh 逐字相同分别是 32 与 30 句，服务本身也有约 2 句的波动。原始输出在 `lora_check/`。

## 负结果：未知领域回退

`unknown_domain_replay.py`：去掉一个领域，给语言模型加“都不合适”选项。去掉 NLP 时 NLP 讲座只有 3.8% 的窗口选“都不合适”，去掉医学时为 0%，模型改选最接近的剩余 slice（`unknown_domain.json`）。多选一的问法不能用来做回退，不进论文。

## 两种部署路由按线上节奏回放

上面的逐块回放拿“带 HMM 的语言模型路由”比“不带规则的原始相似度”，而部署的相似度路由还有确认规则与 pin（run 1 线上覆盖 96.9）。`live_cadence_replay.py` 让两种部署路由读同一批分数（逐块回放存下的余弦分数与字母概率），按线上节奏更新：上下文分数每 5 秒最多刷新一次、文本没变不刷新；相似度路由直接驱动服务端的 `HybridWindowTopicRouter`，参数由闭环实验的同一组命令行参数生成（领先第二名 0.15、领先当前 0.10、连续 3 次、预热 6 秒、冷却 30 秒、pin）；语言模型路由每次刷新走一步 `filter_step`（ρ = 1e-3，证据权重为新块数 / 4）。时间取音频时间。结果 `live_cadence_replay.json`（Taurus CPU）。

校验：按 run 1 自己的发出时间重放相似度路由，19 次切换（线上 19）、前 4 覆盖 96.2（线上 96.9）、确认 slice 正确 89.8（线上 88.5），逐块确认 slice 与线上一致 88.8%。

| 回放（前 4 覆盖，%） | 原始相似度（逐块） | 相似度 + 规则与 pin | 语言模型 + HMM |
| --- | --- | --- | --- |
| 十讲流 | 85.0 | 97.0 | 99.0 |
| 每 15 s 换话题 | 81.7 | 82.1 | 100.0 |
| 每 30 s | 84.9 | 90.5 | 98.2 |
| 每 60 s | 86.8 | 89.3 | 97.9 |
| 每 120 s | 85.8 | 89.6 | 97.4 |
| 每 300 s | 87.1 | 93.5 | 99.3 |
| 德语 | 84.8 | 97.2 | 97.9 |
| 日语 | 84.9 | 97.8 | 99.0 |
| 24 个近领域 slice（术语出现） | 51.6 | 52.2 | 82.7 |

读法：两个领域、讲座很长时，规则与 pin 已把覆盖抬到 97 左右，与语言模型路由只差 1 到 2 点（与 run 3 闭环打平一致）。话题每 15 到 300 秒一换时，规则要连续 3 次确认并冷却 30 秒，新话题升为确认 slice 的中位延迟 27 到 46 秒（60 s 流里 40 段有 12 段始终没确认），覆盖 82 到 94；语言模型路由 97 以上，新话题进前 4 的中位延迟为 0。近领域时规则无济于事（52.2），语言模型路由 82.7。代价：语言模型路由的后验第一名更常换（十讲流 379 次，规则 18 次），十讲流上第一名正确 77.8，规则 91.2；检索用的是前 4，第一名只影响界面显示。

## run 3 闭环结果（十讲流，两张 H200，2026-11-20）

五种配置同一音频顺序、同一切块，打分沿用审计过的流程（`scripts/taurus_run3_score.sh`）。TERM_ACC 是 2,047 个按时间对齐的术语出现，越大越好；配对检验以讲座为单位，精确置换 2^10，讲座 bootstrap 4,000 次。

| 配置 | TERM_ACC NLP / 医学 / 全部 | BLEU | MT-BLEU | 提示精度 | 每块词对 |
| --- | --- | --- | --- | --- | --- |
| AutoTerm-SST（LLM 路由） | 88.4 / 86.3 / 87.7 | 45.10 | 42.27 | 35.7 | 0.94 |
| 相似度路由 | 88.4 / 86.0 / 87.6 | 45.39 | 42.62 | 35.7 | 0.94 |
| Known-domain-1k | 88.3 / 85.9 / 87.5 | 45.15 | 42.41 | 48.7 | 0.69 |
| Merged-2k | 88.8 / 86.0 / 87.9 | 45.31 | 42.48 | 39.1 | 0.86 |
| Merged-10k | 88.6 / 87.0 / 88.1 | 45.11 | 42.33 | 27.1 | 1.26 |

读法：LLM 路由与其余四种都没有显著差别（TERM_ACC 差 −0.4 到 +0.2，p ≥ 0.34；BLEU 差 −0.29 到 −0.01，p ≥ 0.14）。两种路由的线上工作集覆盖 98.0 与 98.7，后验第一名（工作集的第一个 slice）正确 81.5 与 95.8，变化 231 次与 15 次（`run3/routing_coverage.json`）。两领域长讲座太容易，分不出路由高下，所以更难的闭环改用 RealSI（`docs/realsi_20261120/`）。

## Source of Truth

| 产物 | 位置 | revision | 内容与来源 |
| --- | --- | --- | --- |
| 1k topic slices（公开，中性名） | HF dataset `gavinlaw/run-inputs-20261120` | `1d5e87adac9c70606df8b0c611ebc4c4240a11d2` | Taurus `/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/slices/prefix_1k` 的十个 1k slice 与 MaxSim 索引；`stage_slices.py` 把路径改成节点上的 `/data/inputs/slices`，元数据里的项目名换成中性词，向量与术语不变 |
| ACL 60/60 分段 | HF dataset `gavinlaw/rasst-demo-acl6060-zh-segments`（公开） | `a10a6a9bab8758bdc50607b50c62f060ed623109` | 与 Taurus `/mnt/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments` 同一套文件 |
| ESO 医学讲座音频 | HF dataset `gavinlaw/rasst-main-result-data`（公开，早先已发布）`main_result/audio/medicine/` | `96b44797d8926d14ef7535e831b861b6af05a28e` | 五个 wav 的 sha256 与 Taurus `/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine` 一致 |
| 德语、日语 slice 视图（公开） | 同一仓库 `gavinlaw/run-inputs-20261120` | `91ce34aeb3799880c54ff4e7fd455c9c33fffe26` | `build_language_slices.py`：复用英文端向量，目标取已有译法，没有就取 Wikidata 标签（`fetch_wikidata_labels.py`，2026-11-20 抓取）；每个 slice 的德语条数 368–1,000、日语 411–999，education 德语最少（368） |
| 三语 LoRA adapter（公开） | HF model `gavinlaw/rasst-speech-llm-cap16-denoise-ttag-lora` | `a82e03699c697b52a69e5702094fb4a7a81aeb52` | `scripts/extract_lora_from_merged.py`；de 用精确 SVD，zh、ja 用随机化 SVD（与精确 SVD 在 256 个 de 模块上最大相对差 0.34%） |
| run 3 闭环 | 输出 Taurus `/home/jiaxuanluo/autoterm_taurus_20261024_io/runs/r3_*.json`（含 ESO 讲座译文，私有配额已满，状态 `PENDING_HF_UPLOAD`），日志 `runs/r3_logs.tar.gz`；汇总 `run3/*.json` | 代码 `c33dfca` 起的 `scripts/radix_stream_eval/run.sh`，hyper01 两张 H200 | 每个输出的 sha256 在 `run3/results.json` |
| 私有输入副本（未完成） | HF dataset `gavinlaw/autoterm-sst-eval-inputs`（私有） | 只有 ACL 分段 | 私有配额已满（403，要求开自动充值），医学音频与 slice 没传上；闭环任务改用上面三个公开仓库 |

## 坑（在 Radix 节点上跑闭环）

- 框架在 RASST 检索器导入或加载失败时只记一行 ERROR，然后不带检索继续服务（`rag=False`），客户端照常出数，数字看起来正常。`run.sh` 在每个阶段开始前读 `/health`，检索没有 ready 就退出；换环境跑闭环必须保留这一检查。
- 检索器按 `RASST_ACTIVE_CODE_ROOT` 找 RASST 代码，默认是 Taurus 路径；它还按仓库名从 HF 缓存加载 `Atotti/Qwen3-Omni-AudioTransformer`、`BAAI/bge-m3` 和 `openai/whisper-large-v3` 的特征提取配置，离线模式下要先缓存。
- `vllm/vllm-openai:v0.24.0` 只接受 Transformers v5，不能再按旧环境钉 4.57.3；`qwen-omni-utils` 依赖 `audioread`，要显式装。
- ACL 分段的 `segments.meta.jsonl` 与 `audio.yaml` 写的是 Taurus 绝对路径，换机器要改写。

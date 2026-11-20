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

## 负结果：未知领域回退

`unknown_domain_replay.py`：去掉一个领域，给语言模型加“都不合适”选项。去掉 NLP 时 NLP 讲座只有 3.8% 的窗口选“都不合适”，去掉医学时为 0%，模型改选最接近的剩余 slice（`unknown_domain.json`）。多选一的问法不能用来做回退，不进论文。

## Source of Truth

| 产物 | 位置 | revision | 内容与来源 |
| --- | --- | --- | --- |
| 1k topic slices（公开，中性名） | HF dataset `gavinlaw/run-inputs-20261120` | `1d5e87adac9c70606df8b0c611ebc4c4240a11d2` | Taurus `/mnt/data2/jiaxuanluo/autoterm_taurus_20261024/slices/prefix_1k` 的十个 1k slice 与 MaxSim 索引；`stage_slices.py` 把路径改成节点上的 `/data/inputs/slices`，元数据里的项目名换成中性词，向量与术语不变 |
| ACL 60/60 分段 | HF dataset `gavinlaw/rasst-demo-acl6060-zh-segments`（公开） | `a10a6a9bab8758bdc50607b50c62f060ed623109` | 与 Taurus `/mnt/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments` 同一套文件 |
| ESO 医学讲座音频 | HF dataset `gavinlaw/rasst-main-result-data`（公开，早先已发布）`main_result/audio/medicine/` | `96b44797d8926d14ef7535e831b861b6af05a28e` | 五个 wav 的 sha256 与 Taurus `/mnt/data2/jiaxuanluo/RASST/data/main_result/audio/medicine` 一致 |
| 私有输入副本（未完成） | HF dataset `gavinlaw/autoterm-sst-eval-inputs`（私有） | 只有 ACL 分段 | 私有配额已满（403，要求开自动充值），医学音频与 slice 没传上；闭环任务改用上面三个公开仓库 |

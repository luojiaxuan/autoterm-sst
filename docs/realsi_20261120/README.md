# RealSI 十领域闭环测试（2026-11-20）

目的：十讲流只有 NLP 与医学两个领域、讲座又长，两种路由在最终指标上分不出高下。RealSI 的 10 段英译中录音各属一个领域（tech、health、edu、fin、law、env、ent、sci、sport、art，正好对应 10 个 slice），每段 3.4 到 5.2 分钟，合计约 45 分钟。把它们首尾相接，路由要在 10 个领域之间每几分钟切换一次，比十讲流难。

## 数据

- 来源：RealSI（[byteresearchcla/RealSI](https://github.com/byteresearchcla/RealSI)，CC BY 4.0，见 CLASI 论文 arXiv 2407.21646），Taurus 上的副本 `/mnt/data1/jiaxuanluo/rasst_autoterm_10domain/realsi_65f3eff/en2zh/`。音频取自第三方视频，不再分发；节点上用的是下混成 16 kHz 单声道的副本，经本机临时中转。
- 术语标注：RealSI 只给 tech 一段标了术语。`scripts/annotate_realsi_terms.py` 用 `gpt-6.2-astra`（codex-cli 0.170.0，推理档 high）对 10 段统一标注：逐个子句给出领域术语的英文原文片段和参考译文中对应的中文片段，两边都必须逐字出现在该子句里。结果 `realsi_terms.json`，共 439 次出现。与 tech 段的人工标注相比，精度 80.0%（30 次中 24 次与人工一致），召回 92.3%（人工 26 次中找回 24 次）。指引只按 tech 段的第一次试跑改过一次（排除国家、大洲等常见地名，术语限名词短语或专名）。
- 播放顺序：`playlist_s1.json`、`playlist_s2.json`、`playlist_s3.json`，种子 1、2、3 随机打乱 10 段。

## Slice

`scripts/build_realsi_slices.py`：每段讲座的术语（去掉 um、uh 和口吃重复，同一原文取最常见的译法）放在对应领域 1k slice 的最前面，再接原 slice 的条目，跳过原文已被覆盖的，保持 1,000 条；索引用检索器的文本编码器重新编码。Merged-10k 是 10 个新 slice 去掉完全相同的词对后的并集（9,888 条）。各 slice 写入的 RealSI 术语 19 到 46 个。CPU 上试跑时，重新编码的向量与原索引同一词条的余弦相似度最低 0.99992。

## 配置

两种路由都读事先固定的 GPT 生成原型 `configs/router_prototypes_generated_s0.json`：手写的 NLP 原型写了 servers、databases、load balancing，正是 tech 段的内容。语言模型路由用这组原型在十讲流上的离线回放：工作集覆盖 97.9%，手写原型 98.7%（`docs/llm_router_20261120/llm_generated_prototypes.json`）。其余设置与 run 3 相同；Known-domain-1k 按每段讲座切到对应 slice。

## 打分

- TERM_ACC：`eval/streaming_sst/score_realsi_terms.py`，窗口与十讲流相同（子句开始前 2 秒到结束后 30 秒，一次输出只算一次）；参考给出的每种译法都算，括号注释可省。参考译文本身当输出时为 100%（439/439）。
- BLEU：mWERSegmenter 按参考句字符级重切分后算 sacreBLEU（zh），同十讲流；参考译文本身当输出时为 100。
- 配对检验：`eval/streaming_sst/evidence/realsi_eval.py`，以讲座为单位，三种顺序一起交换，精确置换 2^10，讲座 bootstrap 4,000 次。

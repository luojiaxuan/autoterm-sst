# 同目录对照：路由与合并表检索同一批词

论文表 1 中间一块与附录统计表第二块的数字来自这里：AutoTerm-SST 与一张收录完全相同词对的合并表，在 249k、1M 两个目录上比较。

## 目录

| 目录 | 构成 | 合并表词对 | AutoTerm-SST 每块检索 |
| --- | --- | ---: | ---: |
| 249k | 十个话题 slice 各扩到 10,000 条（论文的 1k slice 是它们的前缀），再加 16 个新话题 slice，各约 10,000 条 | 249,218 | 4 万 |
| 1M | 十个 slice 各扩到约 101,000 条 | 1,000,000 | 约 40 万 |

249k 的 16 个话题与原型在 `configs/same_catalog_topics_26.json`；新 slice 取通用 Wikidata 池的前 16 万对，按“术语, 描述”与话题原型的 BGE-M3 相似度在 16 个新话题之间等量分配，原来十个 slice 不动（`scripts/same_catalog/build.py --scale 260k`）。249k 目录与索引公开在 HF dataset `gavinlaw/run-inputs-20261124`，revision `b6d40739fdb676f2b53930ddee9fd4acaaf22da3`。

AutoTerm-SST 用论文的路由器原样（Qwen3-4B 字母概率、粘性 HMM、四个活跃 slice），去掉 4,000 条的上限。测试流与打分口径与论文 run 3 相同（`scripts/same_catalog/score.sh`），COMET 用 `eval/streaming_sst/evidence/comet_quality.py`。1M 一档在三张 A6000 上跑（`run_aries.sh`、`drive_aries.sh`），249k 一档在两张 H100 上跑（`run_radix_arms.sh`、`main_radix_arms.sh`），所以只在同一档内比较。

## 结果

| 目录 | 系统 | NLP | 医学 | TERM_ACC | BLEU | MT-BLEU | COMET | 提示精度 | 每块注入对数 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 249k | AutoTerm-SST | 89.1 | 85.1 | 87.8 | 44.50 | 41.68 | 83.66 | 10.5 | 3.22 |
| 249k | Merged | 87.4 | 83.1 | 85.9 | 43.60 | 40.82 | 83.12 | 4.9 | 6.91 |
| 1M | AutoTerm-SST | 86.7 | 80.4 | 84.6 | 43.26 | 40.38 | 83.30 | 4.4 | 7.35 |
| 1M | Merged | 86.1 | 79.5 | 83.9 | 40.61 | 37.70 | 82.20 | 3.6 | 8.73 |

AutoTerm-SST 减合并表，括号里是讲座级精确置换检验的 p（2^10 种交换），区间是按语料分层的讲座 bootstrap 95% 区间：

| 目录 | TERM_ACC | BLEU | MT-BLEU | COMET |
| --- | --- | --- | --- | --- |
| 249k | +1.86 [+0.63, +3.03]（0.027） | +0.91 [+0.41, +1.41]（0.010） | +0.86 [+0.24, +1.41]（0.031） | +0.54 [+0.30, +0.79]（0.006） |
| 1M | +0.68 [−0.07, +1.43]（0.19） | +2.66 [+0.50, +6.16]（0.037） | +2.69 [+0.48, +6.34]（0.035） | +1.10 [+0.30, +2.85]（0.018） |

249k 下，AutoTerm-SST 与只检索正确 10k slice 的已知领域参考相比，TERM_ACC 差 −0.10（p=0.93），BLEU 差 −0.27（p=0.44），COMET 差 −0.26（p=0.25）。去掉任意一场讲座，AutoTerm-SST-249k 仍领先 Merged-249k 至少 1.30 个 TERM_ACC 点、0.75 BLEU、0.66 MT-BLEU。

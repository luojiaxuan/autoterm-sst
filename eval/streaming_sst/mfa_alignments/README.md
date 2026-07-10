# ACL 与 Medicine 的 MFA 对齐数据

这里保存论文评测所需的轻量级 Montreal Forced Aligner（MFA）标注。文件只包含
Praat TextGrid 时间区间和 ACL 的 alignment diagnostics，不包含音频。评测代码应读取
`words` interval tier，并以秒为单位将每个 source-term occurrence 映射到音频窗口。

## 目录

| 领域 | 内容 | 文件数 | 原始位置（Taurus） |
| --- | --- | ---: | --- |
| ACL 60/60 | 5 个完整 talk 的 TextGrid，以及 `alignment_analysis.csv` | 6 | `/mnt/gemini/data2/jiaxuanluo/acl6060_dev_offline_eval/mfa_textgrids` |
| ESO medicine | 5 个完整 speech 的 TextGrid：404、545006、596001、605000、606 | 5 | `/home/jiaxingxu/rag-sst/eso-dataset/mfa_v1/textgrids` |

ACL talk IDs 为 `2022.acl-long.{110,117,268,367,590}`。Medicine 文件沿用上游
`test_<sample>_full.TextGrid` 命名。数据于 2026-07-09 从 Taurus 原样复制；本目录的
`SHA256SUMS` 与远端源文件逐项一致。

## 完整性检查

在仓库根目录运行：

```bash
shasum -a 256 -c eval/streaming_sst/mfa_alignments/SHA256SUMS
```

## 评分口径

这些时间戳用于构造严格的 occurrence-level 术语指标。若同一术语在 gold 中出现
`k` 次，评分时必须要求输出提供至多 `k` 个可区分的命中（count clipping），不能用
“整段输出出现一次”替代全部 occurrence。旧的 type-level 或 sentence-level
any-hit 指标不能与该口径混用。

原始音频、MFA acoustic model 和大规模派生 chunk 数据不在本目录内；它们仍由各自
数据源和项目 artifact 记录管理。

# AutoTerm 统一评测 Scorecard

`eval/streaming_sst/score_autoterm_story.py` 将同一 playlist 上的
`Oracle`、`AutoTerm` 和 `merged glossary` 三个运行结果汇总成一份 JSON 和一份
Markdown 表。脚本会在评分前强制检查：

1. 三个 JSON 的 blocks 与 block spans 完全一致；
2. 每个持久化翻译事件的 `(start_sample, cursor_samples)` 完全一致；
3. 每个 chunk 捕获的 references 数量与实际 decoder prompt 数量完全一致；
4. headline TERM_ACC 的分母只来自同一份 MFA
   `raw_plus_medicine` occurrence 集合。

主表包含：MFA time-aligned TERM_ACC、BLEU、technical-masked BLEU、
raw-masked BLEU、严格 time-local prompt precision 和 retrieved references per
chunk。旧的 419 分母以及 block-level count-clipping TERM_ACC 不会被计算，也不会
进入输出表。

```bash
python3 eval/streaming_sst/score_autoterm_story.py \
  --oracle /path/to/blockwise_oracle.json \
  --autoterm /path/to/budgeted_autoterm.json \
  --merged /path/to/merged_glossary.json \
  --mfa-root eval/streaming_sst/mfa_alignments \
  --acl-root /path/to/acl6060_zh_segments \
  --acl-reference-text /path/to/acl_zh/ref.txt \
  --acl-technical-gold eval/streaming_sst/acl_gold_technical.json \
  --acl-raw-glossary /path/to/acl6060_tagged_gt_raw_min_norm2.json \
  --medicine-oracle-dir /path/to/medicine_zh \
  --target-lang zh \
  --sacrebleu-tokenizer zh \
  --expected-raw-denominator 1079 \
  --out-json /path/to/autoterm_story_scorecard.json \
  --out-md /path/to/autoterm_story_scorecard.md
```

`--expected-raw-denominator` 是可选的正整数断言。正式四 talk 评测建议显式传入
冻结后的分母；如果 playlist、MFA 数据或 gold inventory 意外变化，脚本会直接
失败。JSON 还记录三个输入文件、playlist、timing windows、MFA gold、reference
和每个 hypothesis 的 SHA-256，便于后续核对。

默认情况下，只要三个系统的 event windows 不完全一致，脚本就会失败。若
stitched Oracle 确实无法产生相同的 event cadence，可显式加入
`--allow-timing-mismatch`。此时脚本只报告各系统的非配对分数，并在 JSON 中写入
`timing_comparable=false`、逐系统 event count、signature 与首个 mismatch；Markdown
会明确禁止 paired delta 或 superiority claim。

# Domain router 离线 replay

`eval/streaming_sst/replay_domain_router_sweep.py` 直接读取
`eval_domain_description_similarity.py` 保存的逐窗口 domain scores，不重新编码
音频，也不运行翻译模型。

它只使用：

- `expected_domain`（gold slice）；
- `scores`（每个 slice 的相似度）；
- 可选的 `text`（仅当 router config 显式选择 `record_text`）。

因此配置选择不会接触 BLEU、TERM_ACC 或其他翻译质量指标。默认预声明规则是：
每个输入的 gold-slice coverage 先达到 99%，再依次最小化 active slices、churn
和 transition delay；若没有配置达到阈值，则先选择最高 coverage，再比较成本。

## 运行

```bash
python eval/streaming_sst/replay_domain_router_sweep.py \
  --input base=/path/to/realsi_domain_description_bgem3_base.json \
  --top-k 1,2,3,4,5,10 \
  --router-configs configs/router_replay_sweep.example.json \
  --min-gold-coverage 0.99 \
  --out-json /path/to/router_sweep.json \
  --out-markdown /path/to/router_sweep.md
```

可重复传入 `--input LABEL=PATH`。多输入选择使用最差输入的 coverage，而不是让
大数据集的 micro average 掩盖某个输入上的失败。JSON 会记录输入 SHA-256、完整
resolved `RouterConfig`、逐配置 decision，以及 Pareto front。

`scores_only` 用一个不含 taxonomy keyword 的中性文本标记满足 production router
的 context-evidence guard；实际 domain evidence 仍只来自保存的 similarity scores。
这条路径在 `HybridWindowTopicRouter` 中使用 `audio_ema_alpha` 更新 EMA，因此 sweep
该参数，而不是 `text_ema_alpha`。若输入记录包含原始窗口文本，可改用
`router_text_mode: record_text`，同时评估 keyword 和 context-similarity evidence。

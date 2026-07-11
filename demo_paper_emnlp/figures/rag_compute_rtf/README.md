# RAG compute RTF 紧凑图

本目录把 RASST PR #1 中冻结的 MaxSim compute 数据重排成适合 AutoTerm-SST
system-demo paper 单栏排版的双面板图。没有修改任何数值。

## 真源

- Repository: `LeiLiLab/RASST`
- PR: <https://github.com/LeiLiLab/RASST/pull/1>
- Commit: `adc47b8b5c0a439d4f4b74cdee02145db520054b`
- Upstream data:
  `docs/results/rag_compute_rtf/data.tsv`
- 用户提供的 upstream PDF SHA-256:
  `ab6f2151195c5130bd3744515c93f5ecbf6ff5c1d1e13ef9e6fc908f5099aff9`

`data.tsv` 保留绘图和审计需要的 frozen 字段。上游图的顶部使用
`rag_mean_rtf_pct`，底部使用 `rag_median_ms`；README 中列出的 median RTF 不能
替代顶部曲线。

## 指标

```text
RAG compute RTF = mean retriever call time / (0.96 s * LM)
```

每次调用编码当前 generation span 与固定 1.92 s lookback。该数据只测量
single-glossary MaxSim retriever，不包含 LLM decoding、AutoTerm context router 或
multi-slice selection。

## 生成

```bash
python demo_paper_emnlp/figures/rag_compute_rtf/render_compact.py
```

输出：

- `demo_paper_emnlp/latex/figures/rag_compute_rtf_compact.pdf`
- `demo_paper_emnlp/latex/figures/rag_compute_rtf_compact.png`

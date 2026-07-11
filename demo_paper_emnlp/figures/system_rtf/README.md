# 标准 system RTF 图

本目录生成正文 Figure 4。两个 panel 都使用标准系统口径，而不是把单个
retriever component time 除以 chunk stride：

```text
system RTF = processing wall time / 输入音频时长
```

左图重新计算 RASST PR #1 commit
`adc47b8b5c0a439d4f4b74cdee02145db520054b` 对应的 Medicine En→Zh LM
sweep。SimulEval 的 `elapsed` 包含 source delay 与 wall processing time，因此每个
talk 使用末条有效 emission 的 `elapsed - delay`，再按全部 source duration 做
micro-average。末条 emission 至 EOF 的未记录尾部只占 0.11%--0.16% audio。
橙色部分是原始 MaxSim timer 在相同 audio denominator 下的 component ratio；
它只是 system RTF 的一部分。

右图使用当前 Hyper H200 单卡 ten-talk AutoTerm 实验。该口径从 WebSocket
connection 计至最终 output cursor，包含 AutoTerm routing、MaxSim retrieval、LLM
generation、scheduler、WebSocket transport 和 closed-loop backpressure；不包含
HTTP session initialization、cold index loading 或最终 teardown。四个 run 均使用同一条
16,848.115 秒、8,776-window 的 ten-talk En→Zh stream，`feed_sleep=0`，最多允许
一个 1.92 秒未确认 chunk。最终 cursor 与全部输入样本一致。

## 输入工件

| Condition | JSON SHA-256 |
| --- | --- |
| Known-domain-1k | `8104be3d333d4a1035f6933d8e3e38d406ae10aa4b1431546fa3a528e1ea8e93` |
| AutoTerm-1k×4 | `3b4fbb01c8d120c432f595bd788b950f263da93105c45e5a32ae9caba632b30f` |
| Merged-100k | `dd03b4f714b177ffa534949c9322e8d39a98c92488673a80aaaf61e54233b58e` |
| Merged-1M | `a3ad72353fac2cabbab031c4d1e307c840929d00ccc56e0a62d97c3e749a0cee` |

完整 JSON 暂存在 Hyper staging：
`/data/autoterm-10talk-budget-20260711/hyper/`，计划上传到
`luojiaxuan/autoterm-sst-10talk-streamlaal-zh`，当前仍为 pending。

`concurrency.tsv` 记录正文段落使用的 2×A6000 vLLM continuous-batching
WebSocket sweep。该指标是 generation p95 / 1.92 秒 stride，不把它称为 system
RTF，也不与两个 panel 的 RTF 混算。

## 生成

```bash
python demo_paper_emnlp/figures/system_rtf/render_system_rtf.py
```

输出：

- `demo_paper_emnlp/latex/figures/system_rtf_scaling_compact.pdf`
- `demo_paper_emnlp/latex/figures/system_rtf_scaling_compact.png`

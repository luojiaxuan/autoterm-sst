# Qwen3-Omni TP=2 并发：sglang-omni vs vLLM 评测小结（#760 跟进）

**我做了什么**
用 ACL6060 dev（en→zh，468 段）搭了一套 SimulEval StreamLAAL/BLEU 评测，在**同一组 2 张 GPU、TP=2** 上对比纯 `Qwen3-Omni-30B-A3B-Instruct`（no-RAG，speech→text）的 sglang-omni 和 vLLM，并发拉到 32 路流式 session。

**结论先行**
- **质量打平**（BLEU ~33，StreamLAAL ~1.33s），差距纯粹在系统吞吐。
- N=1 时 sglang 反而更快（1.10 vs 0.82 seg/s），但**扩到 N=32 被 vLLM 拉开**：vLLM 12.97 vs sglang 8.71 seg/s（约 +49%）。

**各项开销 —— 问题不在之前以为的 GPU-side**
- 真正卡住的是 **host-side 的 per-turn RTT**：N=1→N=32 从 232ms 涨到 822ms，这 3.5x 几乎就是全部 gap；吞吐被死锁在 `32 workers / per-turn-RTT`。
- 逐 stage residency：**thinker 64%，encoder+aggregate 排队 30%，跨进程 relay 只有 ~2%**。
- thinker 跑在 **100% CPU、GPU 只有 60–75%**；共享的 “pipeline” 进程（preprocess + encoder + aggregate + detok + HTTP）被 **GIL 串行**（一个 identity 的 aggregate stage 都能堆 170ms 纯排队）。
- 所有 GPU-side 杠杆我都验证过、都没用：M1 prefill-coalesce、mixed-chunk、thinker overlap，**甚至给 encoder 单独一张卡**（decode 占用 46%→62%，但吞吐纹丝不动）。→ 瓶颈不在 thinker/GPU，relay 也不是，推翻了原帖的假设。

**de-fragmentation 的定位**
prefill 合并确实有用，是个合理的小优化（N=32：8.71→9.41 seg/s），但**它不是追平 vLLM 的那把钥匙**，建议按 “de-fragmentation 小改进” 来定位，别绑在 “对齐 vLLM” 上。

**已提交 3 个 PR，麻烦帮忙 review**
- #770 — 并发 / rollout 压测 runner
- #771 — prefill batching + TP fix（对应 #760）
- #772 — admission control（429 熔断）+ telemetry
完整证据、原始数据、SimulEval agent 和一键复现都放在我 fork 的分支：`benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/`（链接已贴在 #760 评论里）。

**后续优化想再讨论一下**
这个 host-side 开销我倾向认为是 **sglang-omni multi-stage 设计的固有成本**（多进程 + 每 stage 一次 Python hop + GIL），短期很难追上 vLLM 的 mono pipeline。而且我这次压的是 **speech→text 输出，严格说没走 omni 全流程**（没有 talker/code2wav）——真跑全 omni 流程时，多 stage 的分摊收益和结论可能不一样。所以方向（压 thinker per-step CPU 成本 / 给 pipeline 进程 de-GIL 拆进程 / 还是接受这个 trade-off）值得再聊。

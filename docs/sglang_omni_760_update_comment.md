<!-- Ready-to-paste comment for sgl-project/sglang-omni#760. -->
<!-- Layered summary; full evidence + repro live on the fork branch linked below. -->

## Measured update: the TP=2 gap vs vLLM is host-side, not prefill fragmentation

I built a SimulEval StreamLAAL/BLEU harness and A/B'd **stock sglang-omni vs
vLLM** for *pure* `Qwen3-Omni-30B-A3B-Instruct` (no-RAG, en→zh) on the **same 2
GPUs**, TP=2, 32 concurrent streaming sessions. Sharing the results because they
**revise the root-cause hypothesis** in the original post.

**Full evidence, raw data, the SimulEval agent, and one-command repro:**
[`benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/`](https://github.com/luojiaxuan/sglang-omni/tree/diag/qwen3-omni-tp-vllm-gap/benchmarks/diagnostics/qwen3_omni_tp_vllm_gap)
(on my fork).

### 1. The gap (quality at parity)
| engine | N | seg/s | BLEU | StreamLAAL | StreamLAAL_CA |
|--------|---|-------|------|-----------|---------------|
| sglang | 1 | **1.10** | 32.9 | 1326 | 1546 |
| vllm   | 1 | 0.82  | 33.2 | 1340 | 1659 |
| sglang | 32| 8.71  | 32.9 | 1338 | 2144 |
| vllm   | 32| **12.97** | 33.5 | 1328 | 1892 |

sglang is faster single-stream but scales worse (8.7× vs 15.8× over 32×); BLEU /
StreamLAAL parity confirms a fair A/B.

### 2. What does NOT close the gap (all measured, N=32)
| change | seg/s | per-turn RTT | verdict |
|--------|-------|--------------|---------|
| stock | 8.71 | 822 ms | baseline |
| prefill-coalesce (this issue's PR) | 9.41 | — | +marginal, gap intact |
| mixed-chunk | 9.26 | — | null |
| thinker CPU/GPU overlap | 7.76 | 916 ms | **regresses** |
| **dedicated encoder GPU** (3-GPU) | 8.29 | 861 ms | occ 46%→62%, **tput flat** |
| **vLLM** | **12.97** | ~552 ms | target |

Giving the audio encoder its own GPU raised thinker decode-batch occupancy
46%→62% **yet throughput/latency did not move** — so the thinker/GPU is not the
constraint.

### 3. Root cause: host-side per-turn latency
Throughput is rigidly `32 workers / per-turn-RTT`, and RTT is stuck at ~820–860
ms regardless of every GPU-side lever. Per-stage residency (sglang-omni's own
`/start_request_profile` + `python -m sglang_omni.profiler`):

| stage | residency | share |
|-------|-----------|-------|
| thinker (ingest + prefill + decode) | 701 ms | **64%** |
| audio_encoder (86 ms compute + queue) | 170 ms | 15% |
| mm_aggregate (identity → ~all queue) | 170 ms | 15% |
| preprocessing | 34 ms | 3% |
| **all cross-process relay hops** | **~26 ms** | **~2%** |

The thinker runs at **100% CPU with its GPUs at 60–75%** (admission queue is only
10 ms — not a prefill backlog; ~139 ms is the busy loop just ingesting the
request, the rest is decode stretched by per-step Python overhead). The shared
"pipeline" process (preprocess + encoders + aggregate + detok + HTTP for all 32
streams) is GIL-serialized — an identity aggregate stage accruing 170 ms is pure
queueing. vLLM's monolithic engine avoids both.

### 4. Implication
- **Relay-cutting is refuted** (~2% of the path); shm/nccl/nixl backend is not
  the lever (and nccl/nixl won't init on encoder-1GPU + thinker-TP2).
- The prefill-coalesce PR is a legitimate small **de-fragmentation** improvement
  but **not** the vLLM-parity fix — I'd suggest framing it that way.
- The real levers are host/CPU-side and architectural: (a) cut the thinker's
  per-step CPU cost (CUDA-graph decode coverage, leaner step, faster ingest) and
  (b) de-GIL the "pipeline" process (split preprocess/encoder/aggregate/detok
  into separate processes — the speech pipeline config already does this).

Happy to share server logs / the rollout-stress + StreamLAAL JSON on request.

<!-- Evidence companion to docs/sglang_omni_tp_concurrency_issue.md (#760) -->
<!-- Measured follow-up: corrects the prefill-fragmentation hypothesis with data. -->

# Qwen3-Omni TP=2 concurrency: measured vLLM gap and root cause (update to #760)

## TL;DR

I built a SimulEval-native StreamLAAL/BLEU harness and A/B'd **stock sglang-omni
vs vLLM** for *pure* `Qwen3-Omni-30B-A3B-Instruct` (no-RAG, en->zh) on the **same
2 GPUs**, TP=2, max 32 concurrent streaming sessions. Findings:

1. **Quality is at parity** (BLEU ~33, StreamLAAL ~1.33 s both engines). The gap
   is purely systems/throughput.
2. **The gap is real and grows with concurrency:** at N=32 vLLM does **12.97
   seg/s vs sglang 8.71** (vLLM +49%), with lower computation-aware latency.
3. **The gap is NOT prefill fragmentation, and NOT any GPU-side knob.** M1
   prefill-coalesce, mixed-chunk, thinker CPU/GPU overlap, and even giving the
   audio encoder its **own dedicated GPU** are all null on throughput.
4. **Root cause = host-side per-turn latency** in the multi-process pipeline.
   Per-turn round-trip inflates 232 ms (N=1) -> 822 ms (N=32) and that 3.5x is
   100% of the scaling gap. Per-stage profiling: **thinker stage 64%, encoder+
   aggregate queue 30%, cross-process relay only ~2%.** The thinker is CPU-bound
   (100% CPU, GPU 60-75%) and the shared "pipeline" process is GIL-serialized.
   vLLM's monolithic engine avoids both.

So #760's prefill-coalescing work is a legitimate small de-fragmentation PR, but
it is **not** the lever that closes the vLLM gap. The gap is architectural.

## Methodology

- **Harness:** SimulEval streaming agent -> raw OpenAI engine (vLLM `serve` vs
  sglang-omni server), one agent, identical streaming policy. Scored with FBK
  `stream_laal_term.py` (no glossary): BLEU (sacrebleu zh, char), StreamLAAL,
  StreamLAAL_CA. Code: `eval/streaming_sst/` in rasst-demo.
- **Workload:** ACL6060 dev, en->zh, 468 segments, each an independent streaming
  instance; concurrency N = number of parallel SimulEval workers.
- **Model:** pure `Qwen3-Omni-30B-A3B-Instruct` (no-RAG), text output.
- **Hardware:** 2x 48 GB GPUs, TP=2, `mem_fraction_static=0.75`,
  `max_running_requests=32`, `seg=1920 ms`, `max_new_tokens=40`.

## 1. Quantified gap (stock sglang vs vLLM)

| engine | N | seg/s | BLEU | StreamLAAL | StreamLAAL_CA |
|--------|---|-------|------|-----------|---------------|
| vllm   | 1 | 0.82  | 33.2 | 1340 | 1659 |
| vllm   | 32| **12.97** | 33.5 | 1328 | 1892 |
| sglang | 1 | **1.10** | 32.9 | 1326 | 1546 |
| sglang | 32| 8.71  | 32.9 | 1338 | 2144 |

- sglang is **faster single-stream** (N=1: 1.10 vs 0.82 seg/s) but **scales worse**
  (8.7x vs 15.8x over 32x concurrency); crossover ~N=8.
- Quality parity confirms a fair A/B; the gap is systems-only.

## 2. Systematic negative results (what does NOT close the gap)

All at N=32, full 468, same harness:

| change | seg/s | per-turn RTT | verdict |
|--------|-------|--------------|---------|
| stock sglang | 8.71 | 822 ms | baseline |
| M1 prefill-coalesce (#760 PR) | 9.41 | — | +marginal, gap intact |
| mixed-chunk (chunked_prefill) | 9.26 | — | null |
| thinker CPU/GPU overlap (`_event_loop_overlap`) | 7.76 | 916 ms | **regresses** |
| **dedicated encoder GPU** (3-GPU) | 8.29 | 861 ms | occ 46%->62%, **tput flat** |
| dedicated encoder GPU + overlap | 8.40 | 860 ms | null |
| **vLLM** | **12.97** | ~552 ms | target |

Key tell: a dedicated encoder GPU raised thinker decode-batch occupancy from 46%
to 62% **yet throughput and latency did not move** — proving the thinker/GPU is
not the constraint.

## 3. Root cause: host-side per-turn latency

Throughput is rigidly `32 workers / per-turn-RTT`, and RTT is stuck at ~820-860
ms across every config above. Decomposition with sglang-omni's own request-event
profiler (`/start_request_profile` + `python -m sglang_omni.profiler`), N=32:

**Per-stage residency (relative; profiling inflates absolute ms):**

| stage | avg residency | share |
|-------|---------------|-------|
| thinker (ingest + prefill + decode) | 701 ms | **64%** |
| audio_encoder (compute 86 ms + queue) | 170 ms | 15% |
| mm_aggregate (identity -> ~all queue) | 170 ms | 15% |
| preprocessing | 34 ms | 3% |
| **all cross-process relay hops** | **~26 ms** | **~2%** |

**Thinker split:** admission queue is only 10 ms (not a prefill backlog); ~139 ms
is the busy scheduler loop just *ingesting* the request, and the rest is decode
stretched by per-step Python overhead. During load the thinker process sits at
**100% CPU with its GPUs at 60-75%**, and the shared "pipeline" process
(preprocess+encoders+aggregate+detok+HTTP for all 32 streams) is GIL-serialized
(an identity aggregate stage accruing 170 ms is pure queueing).

## 4. Implication / corrected direction

- **Relay-cutting is refuted** (~2% of the path). shm/nccl/nixl relay backend is
  not the lever (nccl/nixl also won't init on the encoder-1GPU + thinker-TP2 map).
- The real levers are host/CPU-side and architectural:
  1. **Thinker scheduler step cost** (100% CPU, GPU idle headroom): broaden
     CUDA-graph decode coverage, leaner per-step Python, faster request ingest.
  2. **De-GIL the "pipeline" process**: split preprocess/encoder/aggregate/detok
     into separate processes (the speech pipeline config already does this) so
     they stop serializing for 32 concurrent streams.
- vLLM is monolithic (one optimized engine, continuous batch, no per-stage Python
  hops), which is the entire ~552 ms vs ~822 ms per-turn-RTT difference.

## Reproduction

```bash
# servers (TP=2, pure Qwen3-Omni-30B-A3B, no-RAG)
GPUS=2,3 PORT=8101 bash eval/streaming_sst/servers/serve_sglang_qwen3omni.sh
bash eval/streaming_sst/servers/serve_vllm_qwen3omni.sh

# concurrency sweep + StreamLAAL/BLEU scoring
bash eval/streaming_sst/run_sweep.sh

# per-stage residency (built-in profiler)
curl -X POST :8101/start_request_profile -H 'Content-Type: application/json' \
  -d '{"run_id":"r","event_dir":"/path/events"}'
#   ... run an N=32 load ...
curl -X POST :8101/stop_request_profile -H 'Content-Type: application/json' -d '{}'
python -m sglang_omni.profiler /path/events --format table
```

Full data, all A/B runs, and analysis: `rasst_eval/runs/COMPARISON.md`.

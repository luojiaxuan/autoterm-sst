# System Scaling: Retrieval & Concurrent Serving

Self-contained record of AutoTerm-SST's systems results — warm-retrieval
scaling and concurrent real-time serving — measured on 2×A6000 (TP=2), 1.92 s
input stride. All numbers are from committed runs.

---

## 1. Retrieval scaling (warm MaxSim)

Warm per-chunk MaxSim retrieval latency stays in a narrow band as the
terminology memory grows four orders of magnitude, because the active slice
queried per chunk stays compact regardless of total inventory size.

| Memory  | Terms | p50 / p95 (ms) | Refs/chunk |
|---------|------:|---------------:|-----------:|
| Curated | 238   | 61.3 / 105.3   | 2.33 |
| Domain  | 10k   | 78.5 / 82.1    | 1.80 |
| Open    | 100k  | 64.1 / 82.6    | 4.89 |
| Stress  | 1M    | 78 / 95        | 9.62 |

**What actually scales is cold index loading**, not per-chunk retrieval:
~0.5 s at 238 terms, 6.8 s at 100k, >30 s at 1M. AutoTerm-SST preloads
candidate working indexes off the streaming batch path, so the streaming loop
never pays that cost. This is why broad memories can serve as fallback pools
while the routed active slice stays small.

---

## 2. Concurrent serving — 32 sessions real-time (verified)

### 2.1 Correction: the earlier "infeasible" read was a unit error

An earlier analysis compared the internal A/B benchmark's **12.97 seg/s**
(vLLM `serve`, N=32, no-RAG, `sglang_omni_vllm_gap_findings.md`) directly
against a "16.67 chunks/s" real-time demand and concluded 32-way was
infeasible. That mixed two units:

- The benchmark's "seg" is a full **ACL segment** — mean 6.64 s = **3.46**
  chunks of 1.92 s (from `segments.meta.jsonl`, 468 segments).
- So 12.97 seg/s = **86 audio-seconds processed per wall-second**.
- 32 real-time sessions demand **32 audio-seconds per wall-second** (each plays
  1 s of audio per 1 s of wall clock).

**86 ÷ 32 = 2.7× headroom.** 32-way real time is feasible; the "12.97 < 16.67"
comparison was segments/s vs chunks/s.

### 2.2 Engine measurement (vLLM continuous batching)

Direct throughput probe against `vllm serve` (V1, continuous batching), TP=2,
2×A6000, unique audio per request (defeats the prefix cache), N=32 saturating:

| CUDA graphs | chunks/s @ N=32 | per-chunk p95 |
|-------------|----------------:|--------------:|
| off (stock `enforce_eager`) | 58   | 0.63 s |
| on          | 63.5 | 0.61 s |

Per-chunk p95 ≈ 0.6 s ≪ 1.92 s stride. CUDA graphs add only ~9 % for
text-only output (decode is 40 tokens), so the stock `enforce_eager` config
already suffices.

### 2.3 Continuous-batching backend + end-to-end result

The in-process offline backend (`vllm.LLM.generate`, single worker, blocking)
serialized inflight batches and degraded to a ~26 s tail-TBT at N=32. A new
per-request backend routes generation to an external `vllm serve` so the vLLM
engine admits and continuously batches concurrent sessions itself:

- `RASST_BACKEND_KIND=vllm_serve` → `VLLMServeBackend` (OpenAI chat,
  `input_audio` base64; history carried as text). Commit `5a0f94d`.

Verified end-to-end through the real framework + WebSocket protocol + demo
model (`concurrency_term_memory.py`, ~real-time feed):

| Concurrent sessions | Partials | Gen p50 | **Gen p95** | < 1.92 s stride |
|--------------------:|---------:|--------:|------------:|:---------------:|
| 8  | 120 | 482 ms | 805 ms | ✓ |
| 16 | 240 | 420 ms | 826 ms | ✓ |
| 24 | 360 | 471 ms | 894 ms | ✓ |
| 32 | 480 | 519 ms | **820 ms** | ✓ |

Per-generation p95 stays flat (~820 ms) from 8→32 sessions while aggregate
throughput scales linearly (1.37→5.47 seg/s) — the continuous-batching
signature — leaving **2.3× margin** under the real-time stride at N=32. (N=1
in a cold run shows ~38 s: first-request engine warmup, not steady state.)

**Conclusion.** 32 concurrent sessions sustain the 1.92 s real-time input rate,
established two independent ways (engine throughput headroom + end-to-end
per-generation latency). The serving engine is stock vLLM and not a
contribution of this work; the paper (§6) states the verified result without a
scaling-limit claim.

---

### Reproduce

```bash
# engine throughput probe (unique audio, N sweep) against a running vllm serve
python throughput_probe.py --n 32 --duration 40 --audio-wav <acl_seg.wav>

# end-to-end framework concurrency (vllm_serve backend)
RASST_BACKEND_KIND=vllm_serve RASST_VLLM_SERVE_URL=http://127.0.0.1:8200 \
  python -m framework.server --port 8025
python eval/streaming_sst/concurrency_term_memory.py \
  --base-url http://127.0.0.1:8025 --preset none --levels 8,16,24,32
```

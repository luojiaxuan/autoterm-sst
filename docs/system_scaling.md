# System Scaling: Retrieval, Routing, and Concurrent Serving

Self-contained record of AutoTerm-SST's systems results — warm-retrieval
scaling, streaming compute RTF, and concurrent real-time serving. Hardware is
reported per section; the frozen RASST RTF package does not record its GPU
model, so no hardware is inferred for that result.

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

## 2. Retrieval and routing compute RTF

RASST PR #1 commit
`adc47b8b5c0a439d4f4b74cdee02145db520054b` freezes the En→Zh Medicine
hard/raw MaxSim compute audit used by Figure 4. The metric is:

```text
retriever compute RTF = retriever call time / (0.96 s * LM)
```

Each call encodes the current generation span plus a fixed 1.92 s lookback.
The upstream figure plots **mean RTF** and **median call time**:

| LM | Cadence | Input span | Calls | Mean call | Median call | Mean RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.96 s | 2.88 s | 14,315 | 53.644 ms | 36.957 ms | 5.5879% |
| 2 | 1.92 s | 3.84 s | 7,159 | 63.977 ms | 42.345 ms | 3.3322% |
| 3 | 2.88 s | 4.80 s | 4,774 | 68.892 ms | 42.560 ms | 2.3921% |
| 4 | 3.84 s | 5.76 s | 3,581 | 58.286 ms | 43.645 ms | 1.5179% |

This package measures the single-glossary MaxSim retriever only; it excludes
LLM decoding, AutoTerm context routing, and multi-slice selection. The compact
paper figure preserves the exact TSV values and is generated from
`demo_paper_emnlp/figures/rag_compute_rtf/`.

The complete ten-talk AutoTerm run separately records the full
`_retrieve_batch` interval, which includes shared speech encoding, BGE-M3
context similarity, router observation/update, and up to four active-index
queries. Across 8,776 windows:

| Stage | Mean | p50 | p95 | Mean / 1.92 s stride |
| --- | ---: | ---: | ---: | ---: |
| AutoTerm routing + four-slice retrieval | 203.272 ms | 204.140 ms | 333.537 ms | 10.587% |

Run SHA-256:
`3b4fbb01c8d120c432f595bd788b950f263da93105c45e5a32ae9caba632b30f`.
The complete JSON remains local staging at
`/data/autoterm-10talk-budget-20260711/hyper/ten_talk_autoterm1kx4_hyper_redundant_audited.json`;
upload to `luojiaxuan/autoterm-sst-10talk-streamlaal-zh` is pending.

---

## 3. Concurrent serving — 32 sessions real-time (verified)

### 3.1 Correction: the earlier "infeasible" read was a unit error

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

### 3.2 Engine measurement (vLLM continuous batching)

Direct throughput probe against `vllm serve` (V1, continuous batching), TP=2,
2×A6000, unique audio per request (defeats the prefix cache), N=32 saturating:

| CUDA graphs | chunks/s @ N=32 | per-chunk p95 |
|-------------|----------------:|--------------:|
| off (stock `enforce_eager`) | 58   | 0.63 s |
| on          | 63.5 | 0.61 s |

Per-chunk p95 ≈ 0.6 s ≪ 1.92 s stride. CUDA graphs add only ~9 % for
text-only output (decode is 40 tokens), so the stock `enforce_eager` config
already suffices.

### 3.3 Continuous-batching backend + end-to-end result

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

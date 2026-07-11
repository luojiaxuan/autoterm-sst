# System Scaling: Retrieval, Routing, and Concurrent Serving

Self-contained record of AutoTerm-SST's systems results — warm-retrieval
scaling, standard system RTF, component timing, and concurrent real-time
serving. Hardware is reported only where it is preserved by the source run.

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

## 2. Standard system RTF

The paper uses the conventional system-level definition:

```text
system RTF = processing wall time / input audio duration
```

### 2.1 RASST LM sweep

RASST PR #1 commit
`adc47b8b5c0a439d4f4b74cdee02145db520054b` freezes the En→Zh Medicine
LM 1--4 runs. For each talk, SimulEval stores source delay and elapsed wall
time at every nonempty target emission. We compute
`last_elapsed - last_delay`, then micro-average over 13,740,783.6875 ms of
source audio:

| LM | Cadence | System RTF | MaxSim/audio | MaxSim/system wall |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.96 s | 0.372130 | 0.055879 | 15.02% |
| 2 | 1.92 s | 0.249610 | 0.033322 | 13.35% |
| 3 | 2.88 s | 0.214872 | 0.023921 | 11.13% |
| 4 | 3.84 s | 0.184844 | 0.015179 | 8.21% |

The original attached plot labeled only `MaxSim time / cadence` as RTF. That
quantity is useful as a component ratio but is not full system RTF. Figure 4
instead stacks the MaxSim component inside the standard total. Because
SimulEval records timing only on nonempty target emissions, unobserved tails
cover 0.11%--0.16% of audio; these values are therefore system RTF to the last
emitted target, not strict EOF-complete RTF. The source package does not record
its GPU model, so none is inferred.

### 2.2 Current ten-talk system RTF

The four current runs use one Hyper H200 and the same 16,848.115 s alternating
stream, retaining a final status event at the complete input cursor. Wall time
starts after session initialization and ends at that final cursor, including
routing, retrieval, generation, scheduler, WebSocket transport, and closed-loop
backpressure. Cold index loading and teardown are excluded.

| Setting | Wall time | System RTF | Throughput |
| --- | ---: | ---: | ---: |
| Known-domain-1k | 6,548.116 s | 0.388656 | 2.573× real time |
| AutoTerm-1k×4 | 6,838.546 s | 0.405894 | 2.464× real time |
| Merged-100k | 6,994.185 s | 0.415132 | 2.409× real time |
| Merged-1M | 7,133.780 s | 0.423417 | 2.362× real time |

AutoTerm's `_retrieve_batch` timer includes shared speech encoding, BGE-M3
context similarity, router observation/update, and up to four active-index
queries. Its 203.272 ms mean is 10.588% of audio duration. This is reported as
a **routing/retrieval stage-time ratio**, not a second system RTF.

The complete AutoTerm JSON has SHA-256
`3b4fbb01c8d120c432f595bd788b950f263da93105c45e5a32ae9caba632b30f`
and remains local staging at
`/data/autoterm-10talk-budget-20260711/hyper/ten_talk_autoterm1kx4_hyper_redundant_audited.json`.
Upload to `luojiaxuan/autoterm-sst-10talk-streamlaal-zh` is pending. Figure
sources and all four run hashes are recorded under
`demo_paper_emnlp/figures/system_rtf/`.

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

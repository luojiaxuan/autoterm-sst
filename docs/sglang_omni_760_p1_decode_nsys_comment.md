<!-- Ready-to-paste comment for sgl-project/sglang-omni#760 (P1 decode-only nsys). -->

## P1: decode-only nsys split — MoE expert GEMM dominates; all-reduce is secondary

Follow-up to P2c. With mixed-chunk off and CUDA graph on, I profiled the **steady
thinker decode forward** (TP=2, N=32 concurrent sessions) and split GPU time
into MoE / all-reduce / dense GEMM / attention / misc / gaps. Goal: decide
whether the next TP thinker lever is **MoE weight traffic** or **TP comm**.

### Setup

| item | value |
|------|-------|
| node | aries, physical GPUs 0,1 (NV4 NVLink pair) |
| model | `Qwen3-Omni-30B-A3B-Instruct` thinker, TP=2 |
| branch | `perf/thinker-decode-opt` (`3e9399f`) — mixed-chunk **off** |
| load | N=32 steady decode, 30 s ramp + 30 s nsys window (job 46210) |
| profiler | nsys session control (`launch → start → stop`), `--cuda-graph-trace=node` |
| isolation | decode-only = kernels with non-null `graphId` in nsys sqlite; dominant steady bucket = `graphId=2` (222 launches, bs≈32, 1065 graph nodes/step) |

Server-side decode stats during the window: **715–756 decode steps**, **100% CUDA
graph hit rate**, decode batch occupancy ≈32.

### 1. Per decode-step summary (bs≈32, `graphId=2`)

| metric | value |
|--------|-------|
| **wall time / step** | **18.6 ms** |
| GPU-busy (union) | 18.2 ms (98%) |
| inter-kernel gap | 0.40 ms (2.1% of wall) |
| concurrency | 88% of wall has ≥2 kernels active (MoE streams overlap) |
| all-reduce kernels / step | 97 (NCCL `RING_LL`, hidden=2048, bs=32 → 128 KB/msg) |

For reference, bs≈24 (`graphId=5`, 150 launches): **16.9 ms/step** wall.

### 2. GPU time breakdown (% of union-busy GPU time)

Categories from kernel names in the decode CUDA graph. Because MoE expert GEMMs
run on ~99 parallel streams, **category durations sum to >100% of wall**; this
table is % of total GPU-busy time (union).

| category | % GPU-busy | inst/step | notes |
|----------|-----------|-----------|-------|
| **MoE expert GEMM** (`fused_moe_kernel`) | **69%** | 96 | dominant; memory-bound on expert weights |
| **all-reduce** (NCCL) | **15%** | 97 | `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` |
| dense GEMMs (qkv / o-proj / lm_head) | 8% | 193 | |
| attention + KV | 2% | 48 | unified `BatchPrefillWithPagedKV` at q_len=1 |
| misc (route / norm / act / rope / copy) | 5% | — | |
| inter-kernel gaps | — | — | 0.40 ms wall (2.1%) |

### 3. Critical-path breakdown (% of wall, sweep-line)

Decomposes **wall time** by counting only intervals where a single category
owns the GPU (exclusive time). Overlap (MoE ∥ all-reduce, etc.) is reported
separately and is not additive with the rows below.

| category | ms/step | % wall |
|----------|---------|--------|
| **MoE expert GEMM** | 11.0 | **59%** |
| all-reduce (exposed on critical path) | 2.4 | 13% |
| dense GEMMs | 1.2 | 6% |
| attention + KV | 0.2 | 1% |
| misc | 0.6 | 3% |
| inter-kernel gaps | 0.4 | 2% |
| overlap (≥2 categories active) | 2.8 | 15% |

All-reduce decomposed further:

| all-reduce component | ms/step |
|---------------------|---------|
| total GPU-time (union) | 4.5 |
| **exposed (critical path)** | **2.4** |
| hidden under MoE compute | 2.1 |

### 4. Decode batch scaling (memory-bound MoE)

| decode bs | wall / step | MoE GEMM (GPU-busy share) |
|-----------|-------------|---------------------------|
| ~24 | 16.9 ms | 69% |
| ~32 | 18.6 ms | 69% |

+33% tokens (24→32) adds only **+9% wall time** → decode MoE is
**memory-bandwidth-bound on expert weight traffic**, not compute-bound.

### 5. custom-AR vs NCCL (same hardware, in-graph microbench)

The captured model decode graph uses **NCCL**, not custom-AR, even on the NVLink
pair (custom-AR is built and enabled by default; routing gap in the capture
path — separate fix). Measured on the same A6000 NVLink pair with CUDA-graph
capture matching the model's decode all-reduce shape (hidden=2048, bs=32):

| path | per all-reduce | per step (×97 AR) |
|------|---------------|-------------------|
| NCCL (in-graph) | 13.3 µs | ~1.29 ms |
| custom-AR (in-graph) | 7.8 µs | ~0.76 ms |

Realistic decode-step delta from enabling custom-AR: **~0.5 ms (~3%)**. Even if
all exposed all-reduce were pure comm with zero overlap, ceiling is **~2.4 ms
(~13%)**.

Eager (non-graph) on the same pair: custom-AR 32 µs vs NCCL 54 µs at bs=32.

### 6. Implication / next step

- Decode GPU forward is **MoE expert GEMM dominated (59–69%)** and
  **memory-bound on expert weight traffic**.
- All-reduce is real but **secondary** on the critical path (~13% wall; true comm
  ~1.3 ms/step in-graph).
- The ~26–30 ms scheduler-side decode step vs **18.6 ms GPU wall** implies
  **~8–11 ms host/scheduling overhead** between graph replays (sampling / detok /
  batch prep) — a separate P2 lever, not the GPU forward itself.

**Next optimization target:** MoE expert weight-traffic reduction / quantization
(FP8 or int8 experts). Planning to validate on **B200** (higher BW + native FP8).
Custom-AR routing fix is a low-effort ~3% side win, not the main lever.

Artifacts: nsys report `decode_tp2_46210.nsys-rep` + sqlite under
`/mnt/taurus/data2/jiaxuanluo/rasst_eval/nsys/`; job script
`aries_p1_nsys_decode.sh`. Happy to share the sweep-line scripts or re-run with
custom-AR forced once the routing gap is patched.

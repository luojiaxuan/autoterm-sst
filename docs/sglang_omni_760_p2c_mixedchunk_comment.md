<!-- Ready-to-paste comment for sgl-project/sglang-omni#760 (P2c update). -->
<!-- Posts the actionable mixed-chunk win + corrects the earlier "mixed-chunk null". -->
<!-- Full evidence + repro on the fork diag branch linked below. -->

## P2c: `--enable-mixed-chunk` closes ~14% of the gap (hardened) — and corrects two earlier calls

Follow-up to the attribution work above. Short version: the streaming-SST thinker
runs **prefill XOR decode** per scheduler step, so it stalls in-flight decodes
whenever it picks a prefill. Folding those decodes into the extend step
(`--enable-mixed-chunk`, already supported, just never enabled) is a **+14–18%
seg/s** win at quality parity — a pure flag flip.

**Full evidence + one-command repro:**
[`benchmarks/diagnostics/qwen3_omni_tp_vllm_gap/`](https://github.com/luojiaxuan/sglang-omni/tree/diag/qwen3-omni-tp-vllm-gap/benchmarks/diagnostics/qwen3_omni_tp_vllm_gap)
(fork, FINDINGS §6).

### 1. Attribution: is the prefill stall a real lever, or arrival-bound?
Instrumented each pure-prefill step to count decode-ready reqs sitting in
`running_batch` that were passed over (`[prefill attribution]`, env-gated).
Steady-state N=32:

- **96.7–98.1% of prefill steps had ready decodes waiting** (mean **~18–20**
  stranded/step); only ~2–3% were genuinely decode-empty.
- `waiting_queue ≈ 0.3` during prefill steps → these are *small single-new-chunk*
  prefills, **not** an admission backlog.

So the workload is **not arrival-bound**: sglang spends ~39% of steps prefill-only
while ~18 sessions sit decode-ready. Fusion is a valid lever.

### 2. The fix: enable mixed-chunk (clean same-node A/B, N=32)
`--enable-mixed-chunk --chunked-prefill-size 8192`, no code change:

| metric (cold / warm) | baseline | mixed-chunk |
|----------------------|----------|-------------|
| seg/s                | 7.05 / 8.21 | **8.30 / 9.37** (+18% / +14%) |
| BLEU                 | 31.9 / 32.3 | 32.9 / 32.8 (parity) |
| StreamLAAL_CA (ms)   | 2468 / 2241 | **2285 / 2115** (−6%) |
| prefill steps stranding decodes | **98.1%** (~20/step) | **0.0%** |

The probe confirms the mechanism: under mixed-chunk `ready_decode_hist={0: …}`
(every prefill step now has an empty `running_batch`) and decode-only steps drop
**1324 → 816** as that work moves into fused steps.

### 3. Hardening (all controls pass)
- **Swapped order** (mixed brought up *first*): mixed 9.28 / 9.26 vs baseline
  7.83 / 8.41 = **+14%** → not an ordering/caching artifact.
- **No low-concurrency regression**: N=1 0.935 vs 0.891 (+5%); N=8 4.314 vs
  4.303 (parity). The win is concentrated where the decode batch is contended.
- **chunked-prefill-size robust**: warm N=32 at 4096 / 8192 / 16384 =
  9.27 / 9.26 / 9.23 seg/s (flat); 8192 is a safe default.

### 4. Two corrections to earlier comments in this thread
- **"mixed-chunk → null" was a measurement error** — it compared 9.26 against an
  inflated warm baseline and called +6% noise. The clean back-to-back A/B + the
  attribution probe (98% → 0% starvation) show it is the one knob that actually
  moves throughput, because it targets the prefill/decode **duty cycle**, not
  occupancy.
- The decode bottleneck is **GPU-forward-bound**, not host-CPU (a
  `cuda.synchronize()` step-phase profile attributed the thinker's "100% CPU" to
  a GPU-sync wait). vs vLLM the differentiator is the **duty cycle**: vLLM fuses
  prefill+decode (~95% of steps advance decode) while stock sglang advanced
  decode on ~58%. Mixed-chunk closes most of that.

**Suggested action:** enable `--enable-mixed-chunk --chunked-prefill-size 8192`
by default for concurrent streaming. Opened as **#789** (defaults the thinker
path; overridable via `server_args_overrides`). I separately verified the
in-code default engages with *no* CLI flag — runtime shows
`chunked_prefill_size=8192` and `[prefill attribution] … 0.0%`. Happy to share
the raw StreamLAAL/BLEU JSON and server logs.

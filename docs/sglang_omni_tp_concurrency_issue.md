<!-- Issue draft for https://github.com/sgl-project/sglang-omni -->
<!-- Filed: https://github.com/sgl-project/sglang-omni/issues/760 -->
<!-- Suggested labels: performance, qwen3-omni, tensor-parallel, scheduler -->

# Title

Qwen3-Omni thinker (TP=2): prefill batches fragment under concurrent streaming audio — large throughput gap vs vLLM

---

## Summary

I'm running a real-time, multi-session **speech-translation** demo on `Qwen3-Omni-30B-A3B` with the disaggregated Qwen3-Omni pipeline. The server starts and inference is correct, but under ~16–32 concurrent streaming sessions the **thinker's prefill scheduling fragments into many tiny prefill batches**, and aggregate throughput / tail latency are much worse than an in-process **vLLM** continuous-batching wrapper on the same 2× A6000. vLLM comfortably handles 32 concurrent sessions; the SGLang-Omni pipeline does not reach the same "sessions per host."

From discussion with maintainers, two points are already clarified:

- `mem_fraction_static` should be left to **auto-tune** (which already reserves encoder headroom via `apply_encoder_mem_reserve`), not hand-tuned.
- Thinker **tensor parallelism (TP)** is an under-invested area and a valid place to optimize.

This issue documents the conditions, a code-level root-cause analysis, a proposed layered fix, and concrete test criteria. I plan to work on a PR.

## Environment

- **GPUs:** 2× NVIDIA RTX A6000 (48 GB). `Qwen3-Omni-30B-A3B` (MoE, ~3B active) does not fit on a single card, so `--thinker-tp-size 2` is required. DP=2 is preferable on large GPUs (e.g. H200), but is not an option at this memory budget; intra-pipeline DP is not available (`dp_size` is effectively 1; "DP=2" means two router workers).
- **Topology:** thinker sharded across both GPUs (TP=2); audio/image encoders colocated on one of the two GPUs (hence a non-default `mem-fraction-static`).
- **Output:** text (translation). Talker / code2wav are not on the hot path.
- **Workload:** long, multi-turn streaming sessions; each session repeatedly prefills small new audio segments → the workload is **prefill-heavy**, not decode-heavy.
- **Branch reproduced on:** `feat/qwen3-omni-rollout-safety` (code paths cited below are present there).

Representative launch (TP=2, encoders colocated):

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path <qwen3-omni-30b-a3b> \
  --model-name qwen3-omni \
  --thinker-tp-size 2 --gpu-thinker-tp 0,1 \
  --gpu-talker 1 --gpu-code2wav 1 \
  --thinker-max-seq-len 8192 \
  --thinker-mem-fraction-static 0.75   # only because the encoder shares the GPU; ideally auto-tuned
```

(My production launcher is a text-output variant that sets `stage.tp_size` on `Qwen3OmniPipelineConfig` directly, since there is no TP-enabled text-output example — see Open Questions.)

## Reproduction

1. Launch the TP=2 server as above.
2. Drive concurrency with the in-repo harness:
   - `python -m benchmarks.eval.qwen3_omni_rollout_stress --base-url http://127.0.0.1:<port> --rollout-counts 1,2,4,8,16,32 --max-tokens 256`
   - streaming TTFT/TTFC: `python benchmarks/eval/benchmark_omni_streaming_ttft.py --base-url http://127.0.0.1:<port> --label tp2 --repeats 5`
3. Compare against a vLLM continuous-batching baseline (`tp_size=2`, `max_num_seqs=32`) on the same 2× A6000.

## Observed vs. expected

- **Observed:** prefill batches stay small (often size 1–2) even when many requests are queued; aggregate `throughput_qps` / `output_throughput` plateau well below the per-session × N ideal; TTFT and decode latency spike under load.
- **Expected (vLLM-like):** with N requests queued, prefill should coalesce toward the `max_running_requests` / `max_prefill_tokens` budget so throughput scales with concurrency until KV/compute saturate.

There is currently no built-in metric for prefill batch size, so part of this work is adding observability (below).

## Pipeline (where fragmentation happens)

```
session audio chunk
  -> preprocessing            (SimpleScheduler)
  -> audio_encoder            (batched: max_batch_size=32, max_batch_wait_ms=50)
  -> mm_aggregate             (identity, max_batch_size=1, no wait)        [A]
  -> thinker (TP=2)           (waiting_queue -> prefill)                   [B][C][D][E][F]
  -> decode loop
```

## Root-cause analysis (code-cited, verified against the current tree)

None of these is the multimodal embedding path: embeddings arrive fully precomputed and `OmniScheduler._is_request_build_ready()` always returns `True` (`sglang_omni/scheduling/omni_scheduler.py:612`), so embedding sync is **not** the blocker.

**A. mm-aggregate fan-in is unbatched.**
`create_aggregate_executor()` returns `SimpleScheduler(_identity)` with no `max_batch_size` / `max_batch_wait_ms` (`sglang_omni/models/qwen3_omni/stages.py:773`). The encoders batch (audio and image encoders both use `max_batch_size=32, max_batch_wait_ms=50`), but the aggregate stage forwards results to the thinker one at a time, so requests trickle into the thinker `waiting_queue`.

**B. No prefill coalesce window.**
The event loop calls `get_next_batch_to_run()` immediately each iteration (`omni_scheduler.py` `_event_loop_normal`), and `PrefillManager.schedule_next_batch()` / `PrefillAdder` greedily prefill whatever is currently queued (`sglang_omni/scheduling/sglang_backend/prefill.py`). An idle thinker therefore prefills a size-1 batch instead of waiting a few ms for the queue to fill — the dominant cause of fragmentation during ramp-up.

**C. Saturation forces one-in/one-out.**
Once the decode batch reaches `max_running_requests`, `num_allocatable_reqs == 0` and no prefill is scheduled until a slot frees, after which slots refill one at a time. Default `max_running_requests=16` (`sglang_omni/scheduling/sglang_backend/server_args_builder.py:16`).

**D. Chunked prefill disabled by default.**
`chunked_prefill_size=None` (`server_args_builder.py:14`), so each (long) audio prefill must complete in a single step and blocks the decode step instead of interleaving; with `max_prefill_tokens=16384` only ~2–3 long-audio prompts fit per prefill step.

**E. Per-loop TP CPU broadcast even when idle.**
For `tp_size>1`, `recv_requests()` → `_recv_scheduler_messages()` calls `broadcast_pyobj()` on every loop iteration, including empty ones (`omni_scheduler.py:474`). This adds a CPU-collective to every scheduler tick and inflates inter-prefill dead time. The leader→follower work fanout also serializes a Python object through an `mp.Queue` (`sglang_omni/scheduling/sglang_backend/tp_control.py`).

**F. TP config/correctness gap (+ CUDA graph).**
`Qwen3OmniPipelineConfig` and its Speech / Colocated variants (`sglang_omni/models/qwen3_omni/config.py:253,274,320`) inherit the base no-op `tensor_parallel_server_args_overrides()` (`sglang_omni/config/schema.py:272`), so they do **not** auto-inject `disable_custom_all_reduce=True` for a TP>1 thinker the way `MingOmniPipelineConfig` does (`sglang_omni/models/ming_omni/config.py:218`). Today it only works because `examples/run_qwen3_omni_speech_server.py:344` injects it manually; a launch through `sglang_omni serve` (which applies `_apply_tensor_parallel_server_args_overrides`, `sglang_omni/cli/serve.py:435`) would not get it. Separately, the multimodal thinker forward returns `can_run_cuda_graph=False` (`sglang_omni/model_runner/thinker_model_runner.py:311`), so every multimodal prefill is a full Python-driven forward.

### Current defaults at issue

| Knob | Location | Default | Effect under concurrency |
| --- | --- | --- | --- |
| `max_running_requests` | `server_args_builder.py:16` | 16 | One-in/one-out refill at saturation (C) |
| `chunked_prefill_size` | `server_args_builder.py:14` | `None` | Long prefills block whole steps (D) |
| `max_prefill_tokens` | `server_args_builder.py:15` | 16384 | ~2–3 long-audio prompts per prefill step |
| mm_aggregate batching | `stages.py:773` | size 1, no wait | Encoder outputs trickle to thinker (A) |
| TP overrides for Qwen3 | `config.py` (inherits `schema.py:272`) | `{}` | No auto `disable_custom_all_reduce` (F) |

## Proposed solution approach (layered)

**Scheduler batching (directly attacks fragmentation):**

- Add a small, configurable **prefill coalesce window** in `OmniScheduler` (hold prefill up to `coalesce_ms` / until queue depth ≥ k when the decode batch is empty or under-full).
- Give `create_aggregate_executor()` a real `max_batch_size` / `max_batch_wait_ms` so encoder outputs reach the thinker as a group.
- Enable a sane `chunked_prefill_size` default for the thinker so long-audio prefills interleave with decode.
- Allow **bulk refill** when multiple decode slots free at once.

**TP efficiency (maintainer-endorsed core):**

- Add `tensor_parallel_server_args_overrides()` to the Qwen3-Omni configs to auto-inject `disable_custom_all_reduce=True` (parity with MingOmni), so non-example launches are correct.
- Gate the per-loop `broadcast_pyobj` on a non-empty inbox (fast empty-path), and/or coalesce control messages before broadcasting.
- Investigate enabling **CUDA graph** for the multimodal forward (at least decode), and re-evaluating the custom all-reduce kernel under the multi-process TP setup.

**Out of scope (defer to maintainers):**

- `mem_fraction_static` auto-tune with a colocated encoder; the PR will not hardcode it.

## Test / acceptance criteria

- **Concurrency sweep** (1, 2, 4, 8, 16, 32) on 2× A6000 TP=2 via `qwen3_omni_rollout_stress.py`, reporting `throughput_qps`, `output_throughput`, `latency_p95_s`, `rtf`, plus streaming TTFT/TTFC from `benchmark_omni_streaming_ttft.py` (metrics defined in `benchmarks/metrics/performance.py`).
- **New observability:** log average prefill batch size and prefill-steps-per-request so fragmentation is measurable before/after.
- **Acceptance:** at 32 concurrent streaming sessions, average prefill batch size and aggregate throughput approach the vLLM TP=2 baseline (target: throughput within a stated % and bounded TTFT p95 / no decode stalls).
- **New CI stage** modeled on the existing stage 11 `tests/test_model/test_qwen3_omni_videoamme_talker_tp2_ci.py`: a TP=2 concurrency throughput test with thresholds derived from measured P95 via `apply_slack()`, added to `.github/workflows/test-qwen3-omni-ci.yaml`.

## Open questions for maintainers

1. Does the `mem_fraction_static` auto-tune already fully account for a colocated audio/image encoder on the same GPU at TP=2 (`apply_encoder_mem_reserve`)? Any recommended `encoder_mem_reserve` for A6000?
2. Is there interest in a TP-enabled **text-output** (translation) example/CLI flag, or should TP keep being configured via `Qwen3OmniPipelineConfig` `stage.tp_size`?
3. Any known blocker to (a) enabling CUDA graph for the multimodal thinker forward, or (b) re-enabling the custom all-reduce kernel under the multi-process TP setup?

---

*Environment to fill before posting:* GPU/driver/CUDA versions, `sglang-omni` commit hash, model checkpoint, and the rollout-stress JSON + server logs (available on request).

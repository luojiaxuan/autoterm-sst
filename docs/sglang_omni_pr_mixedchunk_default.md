# Upstream PR draft: enable mixed-chunk by default for the Qwen3-Omni thinker

`gh` is not available on this host, so create the PR from the browser.

- **Create PR (compare across fork):**
  https://github.com/sgl-project/sglang-omni/compare/main...luojiaxuan:sglang-omni:feat/qwen3-omni-mixed-chunk-default-760?expand=1
- **Base:** `sgl-project/sglang-omni` `main`
- **Head:** `luojiaxuan/sglang-omni` `feat/qwen3-omni-mixed-chunk-default-760`
- **Branch already pushed:** yes (commit `d703d17`)
- **Relates to:** #760

---

## Title

```
feat(qwen3-omni): enable mixed-chunk by default on the thinker path
```

## Body

```markdown
### What

Default `enable_mixed_chunk=True` and `chunked_prefill_size=8192` for the
Qwen3-Omni thinker, set in `create_sglang_thinker_executor_from_config`
(`sglang_omni/models/qwen3_omni/stages.py`). Both stay overridable via
`server_args_overrides` (pass `enable_mixed_chunk=False` to opt out). One file,
+16/-1.

### Why

The thinker scheduler advances **prefill XOR decode** per step. Under concurrent
streaming a large fraction of steps are prefill-only, and any in-flight decodes
just wait for that step to finish. This is a duty-cycle problem, not an
occupancy or arrival problem.

Measured on `Qwen3-Omni-30B-A3B-Instruct`, TP=2, 32 concurrent streaming-SST
sessions (env-gated scheduler probe):

- ~**98%** of pure-prefill steps had decode-ready requests in `running_batch`
  that were passed over (**~18 per step**) — i.e. the workload is *not*
  arrival-bound.
- decode advanced on only ~58% of scheduler iterations.

Mixed-chunk folds those running decodes into the chunk-prefill (extend) step,
restoring the decode duty cycle.

### Result (same-node A/B, en→zh ACL6060 streaming SimulEval)

| metric | baseline | mixed-chunk default | delta |
|---|---|---|---|
| throughput (seg/s) | ~8.1 | ~9.3 | **+14%** |
| StreamLAAL_CA (computation-aware) | ~2280 | ~2143 | **-6%** |
| BLEU | 32.1 | 32.1 | parity |
| pure-prefill steps stranding ready decodes | ~98% | **~0%** | — |

Hardened: holds under swapped A/B ordering (mixed-first), **no regression at
N=1 / N=8**, and robust across `chunked_prefill_size ∈ {4096, 8192, 16384}`
(flat ~9.2–9.3 seg/s; 8192 chosen as a safe default).

### Verifying the default actually engages (no CLI flag)

Launched the thinker with **no** mixed-chunk CLI flags so the *code default* is
the only source of the feature. Runtime confirms:

```
[thinker args] ... chunked_prefill_size=8192 ...
[prefill attribution] prefill_steps=161 with_ready_decodes=0 (0.0%) ...
```

vs. the prior baseline (no default, no flag): `chunked_prefill_size != 8192` and
~96–98% of prefill steps stranding ready decodes.

### Scope / risk

- Affects the **Qwen3-Omni** thinker only. ming-omni has its own
  `create_sglang_thinker_executor_from_config`; the talker is untouched.
- `cli/serve.py` and the qwen3-omni stage `factory_args` do not set these keys,
  so nothing clobbers the default; an explicit `server_args_overrides` (or a
  `--chunked-prefill-size` / `--enable-mixed-chunk` style flag) still wins.
- Mixed-chunk only engages when `chunked_prefill_size > 0`. For short streaming
  prefills the chunk size is a no-op; for very long prefills, chunking at 8192
  is the standard recommended behavior.
```

**Follow-up (correction to my comment above).** I tested the "de-GIL the pipeline
process" lever I suggested — splitting every non-thinker stage into its own OS
process (the topology the speech pipeline already uses, exposed here via a
`--per-stage-processes` flag).

**It regresses.** Within-node A/B on an idle node (pure `Qwen3-Omni-30B-A3B`,
no-RAG, TP=2, N=32, 468 segments, 1850 turns each):

| topology | seg/s | per-turn RTT (p50) | BLEU |
|----------|-------|--------------------|------|
| one `pipeline` process (stock) | **7.46** | **809 ms** | 32.0 |
| per-stage processes (de-GIL split) | 4.06 | 1702 ms | 32.4 |

Splitting **cut throughput 46% and doubled per-turn RTT** at quality parity. It
converts cheap in-process stage handoffs into cross-process relays that serialize
large multimodal payloads (audio features, encoder / merged thinker-input
embeddings) and adds more serial single-threaded stage processes — costing more
than the GIL serialization it removes. (Caveat: client and server shared the node,
so the extra processes add some host-CPU contention; the exact magnitude may be
amplified, but the direction is robust — same client both arms, and the stock
809 ms p50 RTT matches my earlier 822 ms baseline.)

So please **disregard the "de-GIL the pipeline process" suggestion** from my
comment above — the monolithic `pipeline` process is already the better partition
for speech→text, and the host-side cost looks structural (the helpful direction is
*fewer* stage boundaries / more monolithic, like vLLM, not more). The thinker
per-step CPU cost remains the only viable host-side lever I've found.

Updated evidence, the full A/B log, and one-command repro on the fork branch:
[`benchmarks/diagnostics/qwen3_omni_tp_vllm_gap`](https://github.com/luojiaxuan/sglang-omni/tree/diag/qwen3-omni-tp-vllm-gap/benchmarks/diagnostics/qwen3_omni_tp_vllm_gap).

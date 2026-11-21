"""Open-loop replay of both deployed routers at the live update cadence.

On the server, context scores refresh at most once per update interval (5 s) and
only when the routing text has changed. The similarity router then applies its
switch rules: a new slice is confirmed only when its normalized score leads the
runner-up by 0.15 and the confirmed slice by 0.10 in three consecutive updates,
after a 6 s warm-up and not within 30 s of the last switch, and the confirmed slice
stays pinned in the working set. This replay drives the server's own
HybridWindowTopicRouter, configured through serve_glossary_capacity_sweep.py from the
flags of the live runs. The language-model router takes one filter_step of the
sticky HMM per refresh, its letter probabilities weighted by the share of fresh
chunks in the four-chunk context, and its working set is the top four of the
posterior. Both routers read the scores of the per-chunk replays: cosine scores
recomputed here as in router_replay.py, or read from the .cos.npy of
multilingual_router_replay.py and fine_slice_replay.py, and the letter probabilities
saved by llm_router_replay.py, llm_rapid_hmm.py, multilingual_router_replay.py and
fine_slice_replay.py. A chunk's time is the audio time at its end, as in a live
session. --wall-clock-check also replays the similarity router on the ten-talk run's
own emission times and compares it with the live decisions.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(HERE)]

import eval.streaming_sst.serve_glossary_capacity_sweep as sweep  # noqa: E402
from framework.agents.omni import OmniAgent  # noqa: E402
from framework.agents.term_memory.language_model_router import filter_step  # noqa: E402
from framework.agents.term_memory.topic_router import DomainSlice, RouterSessionState  # noqa: E402

CHUNK_S = 1.92
W = 4
RHO = 1e-3
# Chunks between a routing update and the first chunk that sees it (--delay-chunks). The deployed
# router scores in a background worker; one chunk is the bound for a stream fed faster than real time.
DELAY = 0
# The server's clock is far from zero; its guards read a zero timestamp as "never".
T0 = 1000.0
# Router flags of the live similarity-router runs (scripts/radix_stream_eval/realsi.sh, ROUTED).
LIVE_FLAGS = [
    "--port", "0", "--model-path", "unused", "--manifest", "unused", "--rag-model-path", "unused",
    "--tmp-dir", "unused", "--autoterm-policy", "budgeted_top_slices", "--autoterm-default-preset", "nlp_core_1k",
    "--autoterm-term-budget", "4000", "--autoterm-max-active-slices", "4", "--autoterm-context-window-chunks", "4",
    "--autoterm-context-weight", "1.0", "--autoterm-text-weight", "0.0", "--autoterm-domain-probe-weight", "0.0",
    "--autoterm-speech-weight", "0.0", "--autoterm-metadata-weight", "0.0", "--autoterm-ema-alpha", "0.80",
    "--autoterm-update-sec", "5.0", "--autoterm-warmup-sec", "6.0", "--autoterm-cooldown-sec", "30.0",
]


class _Host:
    """The attributes OmniAgent._topic_router_for reads, without loading models."""

    def __init__(self, config, slices):
        self.config = config
        self._topic_routers = {}
        self._slices = slices

    def _catalog(self, language_pair):
        return None

    def _domain_slices_for(self, catalog):
        return self._slices


def build_router(names):
    args = sweep.parse_args(LIVE_FLAGS)
    captured = {}
    agent_cls = sweep.OmniAgent
    sweep.OmniAgent = lambda **kw: captured.setdefault("config", kw["config"])
    try:
        sweep.build_agent(args, manifest=None, presets=[f"{n}_core_1k" for n in names])
    finally:
        sweep.OmniAgent = agent_cls
    # Every slice holds 1k terms, so the 4k budget admits four slices, as in the per-chunk replays.
    slices = [DomainSlice(preset_id=n, domain_id=n, term_count=1000, index_path=n) for n in names]
    host = _Host(captured["config"], slices)
    return OmniAgent._topic_router_for(host, captured["config"].language_pair)


def refreshes(contexts, times, interval):
    """Yields (t, text, refreshed): the routing text at chunk t and whether scores refresh there."""
    text, scored_text, scored_at = "", None, -np.inf
    for t, now in enumerate(times):
        if contexts[t].strip():
            text = contexts[t]
        fresh = bool(text) and (scored_text is None or (text != scored_text and now - scored_at >= interval))
        if fresh:
            scored_text, scored_at = text, now
        yield t, text, fresh


def similarity_rules(names, cos, contexts, times, default):
    """Per chunk, the confirmed slice and the working set of the similarity router."""
    router = build_router(names)
    state = RouterSessionState(active_preset_id=default, active_domain_id=default,
                               created_s=T0, last_decision_s=T0, last_switch_s=T0)
    top, working, scores, pending = [], [], {}, []
    for t, text, fresh in refreshes(contexts, times, float(router.config.update_interval_sec)):
        now = T0 + float(times[t])
        if fresh:
            pending.append((t + DELAY, {n: float(v) for n, v in zip(names, cos[t])}))
        while pending and pending[0][0] <= t:
            scores = pending.pop(0)[1]
        decision = router.observe(state, None, [], now, router_text=text,
                                  router_text_source="generated_target" if text else "none",
                                  domain_probe_scores={}, context_similarity_scores=scores)
        if decision.action == "switch":
            state.active_preset_id = decision.target_preset_id
            state.active_domain_id = decision.target_domain_id
            state.last_switch_s = now
        top.append(state.active_preset_id)
        working.append(list(decision.evidence["slice_selection"]["selected_slice_presets"]))
    return top, working


def language_model_hmm(names, probs, contexts, times, interval=5.0):
    """Per chunk, the posterior's top slice and top four under the live refresh cadence."""
    posterior, last = {}, None
    top, working = [], []
    for t, _, fresh in refreshes(contexts, times, interval):
        if fresh:
            weight = 1.0 if last is None else min(1.0, (t - last) / W)
            posterior = filter_step(posterior, dict(zip(names, map(float, probs[t]))), switch_rate=RHO,
                                    evidence_weight=weight)
            last = t
        ranked = sorted(posterior, key=lambda n: -posterior[n])
        top.append(ranked[0] if ranked else "")
        working.append(ranked[:4])
    if DELAY:
        top = [""] * DELAY + top[:-DELAY]
        working = [[] for _ in range(DELAY)] + working[:-DELAY]
    return top, working


def chunk_metrics(top, working, gold, valid):
    t1 = np.array([a == g for a, g in zip(top, gold)])
    cover = np.array([g in w for w, g in zip(working, gold)])
    changes = sum(a != b for a, b in zip(top[1:], top[:-1]))
    return {"top1_acc": round(float(t1[valid].mean()), 4), "cover@4": round(float(cover[valid].mean()), 4),
            "top1_changes": int(changes)}


def lag_metrics(top, working, gold, pos, n_chunks, valid):
    out = {}
    for label, hit in (("top1", [a == g for a, g in zip(top, gold)]), ("working_set", [g in w for w, g in zip(working, gold)])):
        hit = np.array(hit)
        lag = []
        for start in np.flatnonzero((pos == 0) & valid):
            h = np.flatnonzero(hit[start:start + n_chunks])
            lag.append(h[0] * CHUNK_S if len(h) else np.nan)
        lag = np.array(lag)
        out[f"median_seconds_until_gold_in_{label}"] = round(float(np.nanmedian(lag)), 2) if np.isfinite(lag).any() else None
        out[f"pieces_where_gold_never_in_{label}"] = int(np.isnan(lag).sum())
    return out


def both(names, cos, probs, contexts, times, default):
    return {"similarity_rules": similarity_rules(names, cos, contexts, times, default),
            "language_model_hmm": language_model_hmm(names, probs, contexts, times)}


def encoder(path):
    from transformers import AutoModel, AutoTokenizer
    from router_replay import VARIANTS, centroids, encode

    tok = AutoTokenizer.from_pretrained(path)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(path).eval()
    names, c = centroids(model, tok, VARIANTS["original"], "cpu")

    def cosine(contexts):
        emb = encode(model, tok, [x if x.strip() else " " for x in contexts], "cpu")
        for t in range(1, len(emb)):
            if not contexts[t].strip():
                emb[t] = emb[t - 1]
        return emb @ c.T

    return names, cosine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="the audited ten-talk run JSON")
    ap.add_argument("--encoder", required=True, help="BGE-M3 path or id")
    ap.add_argument("--ten-talk-llm", type=Path, required=True, help="llm_router_replay.py result.w4.npy")
    ap.add_argument("--rapid-llm", type=Path, required=True, help="folder with llm_rapid_hmm.py result.piece<L>.npy")
    ap.add_argument("--pieces", type=int, default=40)
    ap.add_argument("--lang", nargs=5, action="append", default=[],
                    metavar=("LANG", "ACL_LOG", "MED_LOG", "COS_NPY", "LLM_NPY"))
    ap.add_argument("--fine", type=Path, required=True, help="llm_fine_slices.py output")
    ap.add_argument("--fine-cos", type=Path, required=True)
    ap.add_argument("--fine-llm", type=Path, required=True)
    ap.add_argument("--occurrences", type=Path, required=True, help="term_occurrences.jsonl.gz of fine_slice_replay.py")
    ap.add_argument("--wall-clock-check", action="store_true")
    ap.add_argument("--delay-chunks", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    global DELAY
    DELAY = args.delay_chunks
    import llm_router_replay as lrr

    names, cosine = encoder(args.encoder)
    assert names == list(lrr.EN)
    result = {"rho": RHO, "w": W, "flags": LIVE_FLAGS}

    payload = json.load(open(args.run, encoding="utf-8"))
    recs = payload["records"]
    texts = [str(r.get("text") or "") for r in recs]
    gold = [str(r["expected_domain"]).lower() for r in recs]
    contexts, _ = lrr.contexts_for(texts, W)
    cos = cosine(contexts)
    probs = np.load(args.ten_talk_llm)
    assert probs.shape == cos.shape, (probs.shape, cos.shape)
    valid = np.ones(len(recs), dtype=bool)
    valid[0] = False
    audio = [int(r["cursor_samples"]) / 16000 for r in recs]
    routes = both(names, cos, probs, contexts, audio, "nlp")
    result["ten_talk"] = {k: chunk_metrics(*v, gold, valid) for k, v in routes.items()}
    if args.wall_clock_check:
        top, work = similarity_rules(names, cos, contexts, [float(r["emitted_wall_s"]) for r in recs], "nlp")
        live = [str(r["active_preset"]).removesuffix("_core_1k") for r in recs]
        live_work = [[p.removesuffix("_core_1k") for p in (r.get("selected_slice_presets") or [])] for r in recs]
        check = {"replay": chunk_metrics(top, work, gold, valid), "live": chunk_metrics(live, live_work, gold, valid)}
        check["agreement_with_live_confirmed"] = round(float(np.mean([a == b for a, b in zip(top, live)])), 4)
        result["ten_talk_wall_clock_check"] = check
    print(json.dumps({"ten_talk": result["ten_talk"], "check": result.get("ten_talk_wall_clock_check")}), flush=True)

    result["rapid"] = {}
    for piece_s, (rtexts, rgold, pos) in lrr.rapid_streams(recs, payload["block_spans"], args.pieces).items():
        n_chunks = int(pos.max()) + 1
        rctx, _ = lrr.contexts_for(rtexts, W)
        rcos = cosine(rctx)
        rprobs = np.load(args.rapid_llm / f"result.piece{piece_s}.npy")
        assert rprobs.shape == rcos.shape, (piece_s, rprobs.shape, rcos.shape)
        rvalid = np.arange(len(rtexts)) >= n_chunks
        times = [(t + 1) * CHUNK_S for t in range(len(rtexts))]
        row = {}
        for k, (top, work) in both(names, rcos, rprobs, rctx, times, "nlp").items():
            row[k] = chunk_metrics(top, work, rgold, rvalid) | lag_metrics(top, work, rgold, pos, n_chunks, rvalid)
        result["rapid"][f"piece={piece_s}s"] = row
    print(json.dumps({"rapid": result["rapid"]}), flush=True)

    from multilingual_router_replay import chunk_texts, load

    result["multilingual"] = {}
    for lang, acl_path, med_path, cos_path, llm_path in args.lang:
        joiner = "" if lang == "ja" else " "
        acl, med = load(Path(acl_path)), load(Path(med_path))
        mtexts, mgold = [], []
        for i in range(max(len(acl), len(med))):
            for dom, talks in (("nlp", acl), ("medicine", med)):
                if i < len(talks):
                    ct = chunk_texts(talks[i], joiner)
                    mtexts.extend(ct)
                    mgold.extend([dom] * len(ct))
        mctx, _ = lrr.contexts_for(mtexts, W)
        mcos, mprobs = np.load(cos_path), np.load(llm_path)
        assert mcos.shape == mprobs.shape == (len(mtexts), len(names)), (mcos.shape, mprobs.shape, len(mtexts))
        mvalid = np.ones(len(mtexts), dtype=bool)
        mvalid[0] = False
        times = [(t + 1) * CHUNK_S for t in range(len(mtexts))]
        result["multilingual"][lang] = {k: chunk_metrics(*v, mgold, mvalid)
                                        for k, v in both(names, mcos, mprobs, mctx, times, "nlp").items()}
    print(json.dumps({"multilingual": result["multilingual"]}), flush=True)

    from fine_slice_replay import SR, build_slices, norm, term_slices

    fine = json.loads(args.fine.read_text(encoding="utf-8"))
    fnames, _, _ = build_slices(fine)
    owner = term_slices(fine)
    ends = np.array([int(r["cursor_samples"]) / SR for r in recs])
    fcos, fprobs = np.load(args.fine_cos), np.load(args.fine_llm)
    assert fcos.shape == fprobs.shape == (len(recs), len(fnames)), (fcos.shape, fprobs.shape)
    occ_window, occ_slice, occ_domain = [], [], []
    for line in gzip.open(args.occurrences, "rt", encoding="utf-8"):
        o = json.loads(line)
        s = owner.get(norm(o["term"]))
        if s is None:
            continue
        occ_window.append(min(int(np.searchsorted(ends, float(o["t_start"]), side="right")), len(recs) - 1))
        occ_slice.append(s)
        occ_domain.append(o["domain"])
    occ_domain = np.array(occ_domain)
    result["close_domains"] = {"occurrences": len(occ_slice)}
    for k, (_, work) in both(fnames, fcos, fprobs, contexts, ends, fnames[0]).items():
        hit = np.array([s in work[w] for w, s in zip(occ_window, occ_slice)])
        row = {"term_cover@4": round(float(hit.mean()), 4)}
        for d in ("nlp", "medicine"):
            row[f"term_cover@4_{d}"] = round(float(hit[occ_domain == d].mean()), 4)
        result["close_domains"][k] = row
    print(json.dumps({"close_domains": result["close_domains"]}), flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

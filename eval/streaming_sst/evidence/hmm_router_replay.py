"""Open-loop replay of an HMM topic tracker as the AutoTerm router.

The active slice is a hidden state. Each window's raw context similarities
s(d | c_t) (w=4, original prototypes, as in router_replay.py) give the emission
p(obs_t | d) proportional to exp(s(d | c_t) / T); a sticky transition keeps the state
with probability 1 - rho and moves to each other slice with rho / (K - 1). Forward
filtering gives the posterior over slices after every window. A window whose context
text did not change carries no new evidence and only applies the transition.

Working-set policies compared on the same raw scores:
  map_pin    the MAP slice plus the three best other slices by raw score (pin analogue)
  post_top4  the four slices with the highest posterior
  mass       the fewest slices whose posterior mass reaches c, at most four
  mass_fill  ``mass`` topped up to four slices by raw score
Streams: the audited ten-talk run, and the synthetic fast-switching streams of
rapid_switch_replay.py (NLP / medicine pieces of L seconds).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from router_replay import VARIANTS, centroids, encode

CHUNK_S = 1.92
MAX_SLICES = 4
RHOS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
TEMPS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32)
MASSES = (0.9, 0.95, 0.99)
PIECES = (15, 30, 60, 120, 300)


def contexts(texts):
    return ["".join(texts[max(0, t - 4):t]) for t in range(len(texts))]


def rapid_streams(payload, n_pieces):
    recs, spans = payload["records"], payload["block_spans"]
    talks = []
    for s in spans:
        lo, hi = int(s["start_sample"]), int(s["end_sample"])
        talks.append((str(s["expected_domain"]).lower(), [str(r.get("text") or "") for r in recs if lo < int(r["cursor_samples"]) <= hi]))
    streams = {}
    for piece_s in PIECES:
        n_chunks = max(1, round(piece_s / CHUNK_S))
        pools = {"nlp": [], "medicine": []}
        for dom, talk in talks:
            for i in range(0, len(talk) - n_chunks + 1, n_chunks):
                pools[dom].append(talk[i:i + n_chunks])
        texts, gold = [], []
        for p in range(n_pieces):
            dom = "nlp" if p % 2 == 0 else "medicine"
            piece = pools[dom][(p // 2) * 7 % len(pools[dom])]
            texts.extend(piece)
            gold.extend([dom] * len(piece))
        streams[piece_s] = (texts, gold, n_chunks)
    return streams


def embed_contexts(model, tok, ctx, device):
    emb = encode(model, tok, [c if c.strip() else " " for c in ctx], device)
    for t in range(1, len(ctx)):
        if not ctx[t].strip():
            emb[t] = emb[t - 1]
    return emb


def forward_filter(scores, fresh, rho, temp):
    n, k = scores.shape
    post = np.empty_like(scores)
    alpha = np.full(k, 1.0 / k)
    for t in range(n):
        alpha = (1.0 - rho) * alpha + rho / (k - 1) * (1.0 - alpha)
        if fresh[t]:
            logit = scores[t] / temp
            alpha = alpha * np.exp(logit - logit.max())
            alpha /= alpha.sum()
        post[t] = alpha
    return post


def working_sets(post, scores, mass):
    raw_order = np.argsort(-scores, axis=1)
    post_order = np.argsort(-post, axis=1)
    cum = np.cumsum(np.take_along_axis(post, post_order, axis=1), axis=1)
    size = np.minimum(np.argmax(cum >= mass, axis=1) + 1, MAX_SLICES)
    sets = {"map_pin": [], "post_top4": [], "mass": [], "mass_fill": []}
    for t in range(len(post)):
        top = post_order[t, 0]
        sets["map_pin"].append({top, *[d for d in raw_order[t] if d != top][:MAX_SLICES - 1]})
        sets["post_top4"].append(set(post_order[t, :MAX_SLICES]))
        chosen = set(post_order[t, :size[t]])
        sets["mass"].append(chosen)
        sets["mass_fill"].append(chosen | set([d for d in raw_order[t] if d not in chosen][:MAX_SLICES - len(chosen)]))
    return sets


def boundary_latency(top1, gold_idx, starts, ends):
    lags = []
    for b, e in zip(starts, ends):
        hit = np.flatnonzero(top1[b:e] == gold_idx[b])
        lags.append(hit[0] * CHUNK_S if len(hit) else float("nan"))
    lags = np.array(lags)
    return {
        "median_s": round(float(np.nanmedian(lags)), 2) if np.isfinite(lags).any() else None,
        "max_s": round(float(np.nanmax(lags)), 2) if np.isfinite(lags).any() else None,
        "never": int(np.isnan(lags).sum()),
    }


def evaluate(post, scores, gold_idx, valid, starts, ends, mass):
    top1 = post.argmax(1)
    sets = working_sets(post, scores, mass)
    out = {
        "map_top1_acc": round(float((top1[valid] == gold_idx[valid]).mean()), 4),
        "map_switches": int((top1[1:] != top1[:-1]).sum()),
        "map_latency": boundary_latency(top1, gold_idx, starts, ends),
    }
    for name, s in sets.items():
        cover = np.array([g in w for g, w in zip(gold_idx, s)])
        size = np.array([len(w) for w in s])
        out[name] = {"cover": round(float(cover[valid].mean()), 4), "mean_slices": round(float(size[valid].mean()), 3)}
    out["mass"]["single_slice_share"] = round(float((np.array([len(w) for w in sets["mass"]])[valid] == 1).mean()), 4)
    return out


def raw_baseline(scores, gold_idx, valid, starts, ends):
    order = np.argsort(-scores, axis=1)
    rank = np.argmax(order == gold_idx[:, None], axis=1)
    top1 = order[:, 0]
    return {
        "top1_acc": round(float((rank[valid] == 0).mean()), 4),
        "cover@4": round(float((rank[valid] < MAX_SLICES).mean()), 4),
        "top1_switches": int((top1[1:] != top1[:-1]).sum()),
        "top1_latency": boundary_latency(top1, gold_idx, starts, ends),
    }


def sweep(scores, ctx, gold, names, valid, starts, ends):
    gold_idx = np.array([names.index(g) for g in gold])
    fresh = np.array([bool(c.strip()) and (t == 0 or c != ctx[t - 1]) for t, c in enumerate(ctx)])
    result = {"raw": raw_baseline(scores, gold_idx, valid, starts, ends), "fresh_windows": int(fresh.sum()), "grid": []}
    for rho in RHOS:
        for temp in TEMPS:
            post = forward_filter(scores, fresh, rho, temp)
            for mass in MASSES:
                result["grid"].append({"rho": rho, "T": temp, "c": mass, **evaluate(post, scores, gold_idx, valid, starts, ends, mass)})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", type=Path, required=True, help="npz of context embeddings; written on first use")
    ap.add_argument("--pieces", type=int, default=40)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    names, protos = centroids(model, tok, VARIANTS["original"], device)

    payload = json.load(open(args.run, encoding="utf-8"))
    recs = payload["records"]
    real_ctx = contexts([str(r.get("text") or "") for r in recs])
    real_gold = [str(r["expected_domain"]).lower() for r in recs]
    rapid = rapid_streams(payload, args.pieces)
    rapid_ctx = {L: contexts(texts) for L, (texts, _, _) in rapid.items()}

    if args.cache.exists():
        cached = np.load(args.cache)
        real_emb = cached["real"]
        rapid_emb = {L: cached[f"rapid_{L}"] for L in PIECES}
    else:
        real_emb = embed_contexts(model, tok, real_ctx, device)
        rapid_emb = {L: embed_contexts(model, tok, rapid_ctx[L], device) for L in PIECES}
        np.savez_compressed(args.cache, real=real_emb, names=np.array(names), **{f"rapid_{L}": e for L, e in rapid_emb.items()})

    gold_idx = np.array([names.index(g) for g in real_gold])
    n = len(recs)
    real_valid = np.arange(n) >= 1
    real_starts = [t for t in range(1, n) if gold_idx[t] != gold_idx[t - 1]]
    live_top1 = np.array([str(r["active_preset"]).removesuffix("_core_1k") == g for r, g in zip(recs, real_gold)])
    live_cover = np.array([g in {p.removesuffix("_core_1k") for p in (r["selected_slice_presets"] or [])} for r, g in zip(recs, real_gold)])
    result = {
        "run": str(args.run),
        "device": device,
        "grid_axes": {"rho": RHOS, "T": TEMPS, "c": MASSES},
        "real": {
            "windows": int(real_valid.sum()),
            "true_boundaries": len(real_starts),
            "live_heuristic": {
                "top1_acc": round(float(live_top1[real_valid].mean()), 4),
                "cover": round(float(live_cover[real_valid].mean()), 4),
                "switches": int(payload["summary"]["switch_count"]),
                "boundary_latency_s": [round(float(x["latency_s"]), 2) for x in payload["domain_transitions"]],
            },
            **sweep(real_emb @ protos.T, real_ctx, real_gold, names, real_valid, real_starts, real_starts[1:] + [n]),
        },
    }
    for L, (texts, gold, n_chunks) in rapid.items():
        m = len(texts)
        valid = np.arange(m) >= n_chunks
        starts = list(range(n_chunks, m, n_chunks))
        result[f"rapid_{L}s"] = {"windows": int(valid.sum()), **sweep(rapid_emb[L] @ protos.T, rapid_ctx[L], gold, names, valid, starts, [b + n_chunks for b in starts])}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["raw"] if isinstance(v, dict) and "raw" in v else None) for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()

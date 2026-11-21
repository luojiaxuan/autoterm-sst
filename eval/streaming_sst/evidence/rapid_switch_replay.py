"""Open-loop routing replay on synthetic fast-switching streams.

Generated translation chunks of the audited ten-talk AutoTerm run are cut into pieces of
L seconds and interleaved NLP / medicine, so the topic changes every L seconds. Routing
context, prototypes and encoding match router_replay.py. Raw similarity ranking without
switch guards is an upper bound on how quickly the routing signal itself can follow
topic changes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from router_replay import VARIANTS, centroids, encode
from transformers import AutoModel, AutoTokenizer

CHUNK_S = 1.92


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pieces", type=int, default=40)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    names, c = centroids(model, tok, VARIANTS["original"], device)

    payload = json.load(open(args.run, encoding="utf-8"))
    recs = payload["records"]
    spans = payload["block_spans"]
    talks = []
    for s in spans:
        lo, hi = int(s["start_sample"]), int(s["end_sample"])
        talk = [str(r.get("text") or "") for r in recs if lo < int(r["cursor_samples"]) <= hi]
        talks.append((str(s["expected_domain"]).lower(), talk))

    result = {"variant": "original", "w": 4, "pieces": args.pieces}
    for piece_s in (15, 30, 60, 120, 300):
        n_chunks = max(1, round(piece_s / CHUNK_S))
        pools = {"nlp": [], "medicine": []}
        for dom, talk in talks:
            for i in range(0, len(talk) - n_chunks + 1, n_chunks):
                pools[dom].append(talk[i:i + n_chunks])
        texts, gold, pos = [], [], []
        for p in range(args.pieces):
            dom = "nlp" if p % 2 == 0 else "medicine"
            piece = pools[dom][(p // 2) * 7 % len(pools[dom])]
            texts.extend(piece)
            gold.extend([dom] * len(piece))
            pos.extend(range(len(piece)))
        ctx = ["".join(texts[max(0, t - 4):t]) for t in range(len(texts))]
        emb = encode(model, tok, [x if x.strip() else " " for x in ctx], device)
        for t in range(1, len(ctx)):
            if not ctx[t].strip():
                emb[t] = emb[t - 1]
        order = np.argsort(-(emb @ c.T), axis=1)
        gidx = np.array([names.index(g) for g in gold])
        rank = np.argmax(order == gidx[:, None], axis=1)
        pos = np.array(pos)
        valid = np.arange(len(texts)) >= n_chunks
        row = {"chunks_per_piece": n_chunks, "windows": int(valid.sum())}
        for k in (1, 4):
            row[f"cover@{k}"] = round(float((rank[valid] < k).mean()), 4)
        for k in (1, 4):
            lag = []
            for start in np.flatnonzero((pos == 0) & valid):
                hit = np.flatnonzero(rank[start:start + n_chunks] < k)
                lag.append(hit[0] * CHUNK_S if len(hit) else float("nan"))
            lag = np.array(lag)
            row[f"median_seconds_until_gold_in_top{k}"] = round(float(np.nanmedian(lag)), 2) if np.isfinite(lag).any() else None
            row[f"pieces_where_gold_never_reaches_top{k}"] = int(np.isnan(lag).sum())
        result[f"piece={piece_s}s"] = row
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

"""Rapid topic switches under the language-model router with and without the HMM filter.

The NLP and medicine talks of a saved run are cut into pieces of 15 to 300 seconds and
interleaved into 40-piece streams, as in llm_router_replay.py. Each stream's per-window
letter probabilities are saved, and switch metrics are reported for the raw scores and
for the sticky HMM posterior over the same grid of switch rates as llm_hmm_filter.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llm_hmm_filter import forward
from llm_router_replay import (
    DESCRIPTIONS,
    EN,
    LetterScorer,
    carry_forward,
    contexts_for,
    rapid_streams,
    switch_metrics,
)

RHOS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pieces", type=int, default=40)
    args = ap.parse_args()
    scorer = LetterScorer(args.model, "cuda" if torch.cuda.is_available() else "cpu", args.batch)

    payload = json.load(open(args.run, encoding="utf-8"))
    names = list(EN)
    descriptions = {n: f"{DESCRIPTIONS[n]} {EN[n]}" for n in names}
    result = {"model": args.model, "w": args.w, "pieces": args.pieces, "streams": {}}
    for piece_s, (texts, gold, pos) in rapid_streams(payload["records"], payload["block_spans"], args.pieces).items():
        npy = args.out.with_name(f"{args.out.stem}.piece{piece_s}.npy")
        if npy.exists():
            scores = np.load(npy)
        else:
            ctx, nonempty = contexts_for(texts, args.w)
            scores = carry_forward(scorer.score([c if c.strip() else "(none)" for c in ctx], names, descriptions), nonempty)
            np.save(npy, scores)
        row = {"raw": switch_metrics(scores, names, gold, pos)}
        for rho in RHOS:
            row[f"rho={rho:g}"] = switch_metrics(forward(scores, rho, 1.0 / args.w), names, gold, pos)
        result["streams"][f"piece={piece_s}s"] = row
        print(json.dumps({f"piece={piece_s}s": row}), flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

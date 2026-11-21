"""Language-model router on the ten-talk run with generated slice prototypes.

Same replay as llm_router_replay.py at w=4, except that each slice is described by a
generated prototype (the description line and the English topic sentence of a file
written by scripts/term_memory/generate_router_prototypes.py) instead of the
hand-written DESCRIPTIONS and EN texts. Raw scores and the sticky HMM posterior over
the llm_hmm_filter.py grid of switch rates are reported.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llm_hmm_filter import forward
from llm_router_replay import EN, LetterScorer, carry_forward, contexts_for
from routing_metrics import ranking_metrics

RHOS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--prototypes", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    generated = json.load(open(args.prototypes, encoding="utf-8"))
    names = list(EN)
    descriptions = {n: f"{generated[n][0]} {generated[n][1]}" for n in names}
    recs = json.load(open(args.run, encoding="utf-8"))["records"]
    gold = [str(r["expected_domain"]).lower() for r in recs]
    texts = [str(r.get("text") or "") for r in recs]
    valid = np.ones(len(recs), dtype=bool)
    valid[0] = False

    npy = args.out.with_suffix(f".w{args.w}.npy")
    if npy.exists():
        scores = np.load(npy)
    else:
        scorer = LetterScorer(args.model, "cuda" if torch.cuda.is_available() else "cpu", args.batch)
        ctx, nonempty = contexts_for(texts, args.w)
        scores = carry_forward(scorer.score([c if c.strip() else "(none)" for c in ctx], names, descriptions), nonempty)
        np.save(npy, scores)

    result = {"model": args.model, "prototypes": str(args.prototypes), "w": args.w, "descriptions": descriptions,
              "raw": ranking_metrics(scores, names, gold, valid, ks=(1, 4)), "filtered": {}}
    for rho in RHOS:
        post = forward(scores, rho, 1.0 / args.w)
        result["filtered"][f"rho={rho:g}"] = ranking_metrics(post, names, gold, valid, ks=(1, 4))
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "descriptions"}, indent=1))


if __name__ == "__main__":
    main()

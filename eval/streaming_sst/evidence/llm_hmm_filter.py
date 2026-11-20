"""Sticky HMM filtering of per-window router probabilities.

The hidden state is the topic of the talk. The observation at window t is the
router's probability over slices, treated as a likelihood up to a constant. The
topic stays with probability 1 - rho per chunk and otherwise moves to one of the
other slices uniformly. Consecutive windows share w - 1 of their w context chunks,
so each window's probability enters with exponent 1 / w, which counts every chunk's
evidence once. The working set is the four slices with the highest filtered
posterior. The only free parameter is rho, swept here on a log grid.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from routing_metrics import DOMAINS, ranking_metrics


def forward(probs, rho, beta, eps=1e-6):
    n, k = probs.shape
    loglik = beta * np.log(np.clip(probs, eps, None))
    post = np.full(k, 1.0 / k)
    out = np.empty_like(probs)
    stay, move = 1.0 - rho, rho / (k - 1)
    for t in range(n):
        prior = stay * post + move * (1.0 - post)
        logp = np.log(prior) + loglik[t]
        logp -= logp.max()
        post = np.exp(logp)
        post /= post.sum()
        out[t] = post
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True, help="per-window probabilities saved by llm_router_replay.py")
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = json.load(open(args.run, encoding="utf-8"))["records"]
    gold = [str(r["expected_domain"]).lower() for r in recs]
    probs = np.load(args.scores)
    names = list(DOMAINS)
    valid = np.ones(len(recs), dtype=bool)
    valid[0] = False

    result = {"beta": 1.0 / args.w, "raw": ranking_metrics(probs, names, gold, valid, ks=(1, 4)), "filtered": {}}
    for rho in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        post = forward(probs, rho, 1.0 / args.w)
        result["filtered"][f"rho={rho:g}"] = ranking_metrics(post, names, gold, valid, ks=(1, 4))
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

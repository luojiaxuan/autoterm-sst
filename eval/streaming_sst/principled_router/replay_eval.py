"""Open-loop replay of the principled router on an audited run, against the live router.

Metrics per window (window 0 excluded): coverage = gold-domain slice loaded; mean loaded slices;
set changes; unknown posterior mass. Hyperparameters are chosen by leave-one-pair-out CV over the
five (ACL, medicine) talk pairs so no reported test number is tuned on its own talks.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from hmm_router import RouterParams, filter_posteriors, select_slices

DOMAINS = ["nlp", "medicine", "education", "finance", "legal", "environment", "entertainment", "science", "sports", "art"]


def prototypes(z, prefix, names):
    pn, owner, emb = list(z["proto_names"]), z["proto_owner"], z["proto_text_emb"]
    c = np.stack([emb[owner == pn.index(f"{prefix}:{d}")].mean(0) for d in names])
    return c / np.linalg.norm(c, axis=1, keepdims=True)


def metrics(mask, post, gold_idx, valid):
    cov = mask[np.arange(len(mask)), gold_idx]
    changes = int((mask[1:] != mask[:-1]).any(axis=1).sum())
    return {"coverage": float(cov[valid].mean()), "mean_slices": float(mask[valid].sum(1).mean()),
            "set_changes": changes, "unknown_mass": float(post[valid, -1].mean()) if post is not None else None,
            "empty_frac": float((mask[valid].sum(1) == 0).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    z = np.load(args.emb, allow_pickle=False)
    gold = z["gold"]
    cursor = z["cursor_s"]
    n = len(gold)
    gold_idx = np.array([DOMAINS.index(g) for g in gold])
    # note (luojiaxuan): talk pairs = consecutive (ACL, medicine) blocks; block boundaries are gold-domain changes.
    block = np.concatenate([[0], np.cumsum(gold[1:] != gold[:-1])])
    pair = block // 2
    valid_all = np.arange(n) > 0
    res = {"windows": n, "pairs": int(pair.max() + 1)}

    live = np.array([[f"{d}_core_1k" in s for d in DOMAINS] for s in map(json.loads, z["live_slices"])])
    res["live"] = metrics(live, None, gold_idx, valid_all)
    res["static_nlp_med"] = {"coverage": 1.0, "mean_slices": 2.0, "set_changes": 0}

    proto = prototypes(z, "coarse", DOMAINS)
    grid = {"window": [1, 2, 4, 8], "kappa": [10, 20, 40, 80, 160], "hazard": [1 / 30, 1 / 120, 1 / 600, 1 / 3000],
            "c_unknown": [0.30, 0.35, 0.40, 0.45], "tau": [0.01, 0.05, 0.2]}
    rows = []
    for w, kappa, hz, cu, tau in itertools.product(*grid.values()):
        p = RouterParams(kappa=kappa, hazard=hz, c_unknown=cu, tau=tau, budget=4, window=w)
        sim = z[f"ctx_w{w}"] @ proto.T
        post = filter_posteriors(sim, cursor, z[f"nonempty_w{w}"], p)
        mask = select_slices(post, p)
        cov = mask[np.arange(n), gold_idx]
        per_pair = [(float(cov[(pair == q) & valid_all].mean()), float(mask[(pair == q) & valid_all].sum(1).mean())) for q in range(res["pairs"])]
        rows.append({"params": p.__dict__, "all": metrics(mask, post, gold_idx, valid_all), "per_pair": per_pair})
    # note (luojiaxuan): CV objective = coverage minus 0.05 per loaded slice (a slice must buy >=5 points of coverage).
    lam = 0.05
    cv = []
    for q in range(res["pairs"]):
        def score(r, held):
            return np.mean([c - lam * s for i, (c, s) in enumerate(r["per_pair"]) if (i == q) == held])
        best = max(rows, key=lambda r: score(r, held=False))
        cv.append({"held_pair": q, "params": best["params"], "test_coverage": best["per_pair"][q][0], "test_mean_slices": best["per_pair"][q][1]})
    res["cv"] = cv
    res["cv_mean"] = {"coverage": float(np.mean([c["test_coverage"] for c in cv])), "mean_slices": float(np.mean([c["test_mean_slices"] for c in cv]))}
    res["grid"] = rows
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("live", "static_nlp_med", "cv", "cv_mean")}, indent=1))


if __name__ == "__main__":
    main()

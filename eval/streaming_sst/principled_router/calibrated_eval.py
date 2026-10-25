"""Per-slice calibrated emissions for the HMM router, with three score functions.

Every score f_k(e) is standardised on held-out pseudo-contexts of slice k itself:
z_k = (f_k(e) - m_k) / s_k. The HMM then uses log p(e | k) = kappa * z_k and
log p(e | unknown) = kappa * z0, so z0 is a tail position of each slice's own score distribution
(z0 = -2 means "farther from every slice than ~98% of that slice's own contexts"), not a raw
cosine. Score functions: desc = cosine to the hand-written description prototype; data = cosine to
the mean of the slice's training pseudo-contexts; logreg = multinomial logistic-regression logit
trained on the pseudo-contexts (C chosen by cross-validation). Pseudo-contexts come from
embed_pseudo.py; no evaluation-talk term is used for fitting or calibration.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from hmm_router import RouterParams, filter_posteriors, select_slices
from replay_eval import DOMAINS, prototypes
from stress_eval import BASE


def fit_scorers(pseudo_emb, pseudo_lab, desc_proto, names, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pseudo_lab))
    keep = np.isin(pseudo_lab, names)
    tr = idx[: len(idx) // 2][keep[idx[: len(idx) // 2]]]
    ca = idx[len(idx) // 2:][keep[idx[len(idx) // 2:]]]
    y_tr = np.array([names.index(l) for l in pseudo_lab[tr]])
    y_ca = np.array([names.index(l) for l in pseudo_lab[ca]])
    data_proto = np.stack([pseudo_emb[tr][y_tr == k].mean(0) for k in range(len(names))])
    data_proto /= np.linalg.norm(data_proto, axis=1, keepdims=True)
    clf = LogisticRegressionCV(Cs=8, cv=5, max_iter=2000).fit(pseudo_emb[tr], y_tr)
    raw = {"desc": lambda e: e @ desc_proto.T, "data": lambda e: e @ data_proto.T, "logreg": clf.decision_function}
    scorers = {}
    for name, f in raw.items():
        s = f(pseudo_emb[ca])
        m = np.array([s[y_ca == k, k].mean() for k in range(len(names))])
        sd = np.array([s[y_ca == k, k].std() for k in range(len(names))])
        scorers[name] = (lambda e, f=f, m=m, sd=sd: (f(e) - m) / sd)
    return scorers, {"logreg_C": float(clf.C_[0]), "pseudo_heldout_acc": float((clf.predict(pseudo_emb[ca]) == y_ca).mean())}


def route(z_scores, z, p):
    post = filter_posteriors(z_scores, z["cursor_s"], z[f"nonempty_w{p.window}"], p)
    return post, select_slices(post, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--pseudo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    z = np.load(args.emb)
    ps = np.load(args.pseudo)
    gold = z["gold"]
    valid = np.arange(len(gold)) > 0
    ctx = z[f"ctx_w{BASE.window}"]
    kappas, z0s = [0.5, 1, 2, 4, 8], [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0]
    res = {"base": {k: v for k, v in BASE.__dict__.items() if k not in ("kappa", "c_unknown")}, "kappa_grid": kappas, "z0_grid": z0s}

    for setting in ("full", "held_nlp", "held_medicine"):
        held = None if setting == "full" else setting.removeprefix("held_")
        names = [d for d in DOMAINS if d != held]
        scorers, info = fit_scorers(ps["emb"], ps["label"], prototypes(z, "coarse", names), names, seed=0)
        res.setdefault("fit", {})[setting] = info
        kn = valid if held is None else (gold != held) & valid
        gi = np.array([names.index(g) for g in gold[kn]])
        out = {}
        for sname, f in scorers.items():
            zs = f(ctx)
            rows = []
            for kappa in kappas:
                for z0 in z0s:
                    p = RouterParams(**{**BASE.__dict__, "kappa": kappa, "c_unknown": z0})
                    post, m = route(zs, z, p)
                    row = {"kappa": kappa, "z0": z0, "known_coverage": round(float(m[kn][np.arange(kn.sum()), gi].mean()), 4),
                           "mean_slices": round(float(m[kn].sum(1).mean()), 3), "set_changes": int((m[1:] != m[:-1]).any(axis=1).sum())}
                    if held is not None:
                        row["heldout_reject"] = round(float((m[(gold == held) & valid].sum(1) == 0).mean()), 4)
                    rows.append(row)
            out[sname] = rows
        res[setting] = out
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["fit"]))


if __name__ == "__main__":
    main()

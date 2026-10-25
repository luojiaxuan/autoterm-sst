"""Trade-off curve of the unknown reference c_u: coverage on known-domain windows vs rejection on held-out-domain windows."""

import argparse
import json
from pathlib import Path

import numpy as np

from hmm_router import RouterParams
from replay_eval import DOMAINS, prototypes
from stress_eval import BASE, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    z = np.load(args.emb)
    gold = z["gold"]
    valid = np.arange(len(gold)) > 0
    res = {}
    for held in ("nlp", "medicine"):
        keep = [d for d in DOMAINS if d != held]
        pr = prototypes(z, "coarse", keep)
        rows = []
        for cu in np.round(np.arange(0.40, 0.561, 0.01), 2):
            pst, m = run(z, pr, RouterParams(**{**BASE.__dict__, "c_unknown": float(cu)}))
            kn = (gold != held) & valid
            gi = np.array([keep.index(g) for g in gold[kn]])
            rows.append({"c_unknown": float(cu), "known_coverage": round(float(m[kn][np.arange(kn.sum()), gi].mean()), 4),
                         "heldout_reject": round(float((m[(gold == held) & valid].sum(1) == 0).mean()), 4)})
        res[f"held_{held}"] = rows
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        print(k)
        for r in v:
            print("  ", r)


if __name__ == "__main__":
    main()

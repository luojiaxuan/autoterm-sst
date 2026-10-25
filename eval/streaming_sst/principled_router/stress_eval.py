"""Close-domain, unknown-domain and one-at-a-time sweep replays for the principled router.

Close domain: 24 prototypes (8 NLP subfields, 8 medicine subspecialties, 8 other coarse domains);
gold is the talk-level subfield label in SUBFIELD_GOLD. Unknown domain: the gold domain's
prototype is removed from the catalog, so every loaded slice in those windows is wrong; we report
how often the router loads nothing (unknown state wins) versus a raw top-B cosine router.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from hmm_router import RouterParams, filter_posteriors, select_slices
from replay_eval import DOMAINS, prototypes

# note (luojiaxuan): talk-level labels from paper titles (ACL Anthology) and ESO talk topics; one label per talk.
SUBFIELD_GOLD = [
    "nlp:dialogue", "medicine:surgery", "nlp:question_answering", "medicine:oncology", "nlp:syntax_semantics",
    "medicine:oncology", "nlp:language_models", "medicine:radiation_therapy", "nlp:language_models", "medicine:oncology",
]
BLOCK_ITEMS = ["2022.acl-long.268", "medicine_404", "2022.acl-long.367", "medicine_606", "2022.acl-long.590",
               "medicine_545006", "2022.acl-long.110", "medicine_596001", "2022.acl-long.117", "medicine_605000"]
ITEM_LABEL = {"2022.acl-long.268": "nlp:syntax_semantics", "2022.acl-long.367": "nlp:language_models",
              "2022.acl-long.590": "nlp:language_models", "2022.acl-long.110": "nlp:dialogue",
              "2022.acl-long.117": "nlp:question_answering", "medicine_404": "medicine:surgery",
              "medicine_606": "medicine:oncology", "medicine_545006": "medicine:oncology",
              "medicine_596001": "medicine:radiation_therapy", "medicine_605000": "medicine:oncology"}
BASE = RouterParams(kappa=20, hazard=1 / 600, c_unknown=0.45, tau=0.01, budget=4, window=8)


def run(z, proto, p):
    sim = z[f"ctx_w{p.window}"] @ proto.T
    post = filter_posteriors(sim, z["cursor_s"], z[f"nonempty_w{p.window}"], p)
    return post, select_slices(post, p)


def raw_topb(z, proto, w, b):
    sim = z[f"ctx_w{w}"] @ proto.T
    order = np.argsort(-sim, axis=1)[:, :b]
    m = np.zeros_like(sim, dtype=bool)
    m[np.arange(len(sim))[:, None], order] = True
    return m


def summarize(mask, gold_idx, valid):
    cov = mask[np.arange(len(mask)), gold_idx]
    return {"coverage": round(float(cov[valid].mean()), 4), "mean_slices": round(float(mask[valid].sum(1).mean()), 3),
            "set_changes": int((mask[1:] != mask[:-1]).any(axis=1).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    z = np.load(args.emb)
    gold = z["gold"]
    n = len(gold)
    valid = np.arange(n) > 0
    block = np.concatenate([[0], np.cumsum(gold[1:] != gold[:-1])])
    res = {"base_params": BASE.__dict__}

    proto = prototypes(z, "coarse", DOMAINS)
    gidx = np.array([DOMAINS.index(g) for g in gold])
    post, mask = run(z, proto, BASE)
    res["coarse_base"] = summarize(mask, gidx, valid)

    # note (luojiaxuan): one-at-a-time sweeps around the CV-selected point.
    sweeps = {"kappa": [5, 10, 20, 40, 80, 160], "hazard": [1 / 15, 1 / 60, 1 / 300, 1 / 600, 1 / 1800, 1 / 6000],
              "c_unknown": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55], "tau": [0.001, 0.01, 0.05, 0.1, 0.2, 0.4], "window": [1, 2, 4, 8]}
    res["sweeps"] = {}
    for name, vals in sweeps.items():
        rows = []
        for v in vals:
            p = RouterParams(**{**BASE.__dict__, name: v})
            pst, m = run(z, proto, p)
            rows.append({name: v, **summarize(m, gidx, valid), "unknown_mass": round(float(pst[valid, -1].mean()), 4)})
        res["sweeps"][name] = rows

    # note (luojiaxuan): unknown domain, one held-out coarse domain at a time.
    unk = {}
    for held in ("nlp", "medicine"):
        keep = [d for d in DOMAINS if d != held]
        pr = prototypes(z, "coarse", keep)
        pst, m = run(z, pr, BASE)
        sel = (gold == held) & valid
        raw = raw_topb(z, pr, BASE.window, BASE.budget)
        unk[held] = {"windows": int(sel.sum()), "hmm_load_nothing": round(float((m[sel].sum(1) == 0).mean()), 4),
                     "hmm_mean_wrong_slices": round(float(m[sel].sum(1).mean()), 3),
                     "hmm_unknown_mass": round(float(pst[sel, -1].mean()), 4),
                     "raw_top4_mean_wrong_slices": round(float(raw[sel].sum(1).mean()), 3),
                     "other_domain_coverage_hmm": round(float(m[(gold != held) & valid][np.arange(((gold != held) & valid).sum()), [keep.index(g) for g in gold[(gold != held) & valid]]].mean()), 4)}
    res["unknown_domain"] = unk

    # note (luojiaxuan): close domain with subfield prototypes.
    pn = [s.removeprefix("fine:") for s in z["proto_names"] if s.startswith("fine:")]
    fine_names = pn + [f"{d}:{d}" for d in DOMAINS if d not in ("nlp", "medicine")]
    owner, emb, allnames = z["proto_owner"], z["proto_text_emb"], list(z["proto_names"])
    rows = []
    for nm in fine_names:
        key = f"fine:{nm}" if f"fine:{nm}" in allnames else f"coarse:{nm.split(':')[0]}"
        rows.append(emb[owner == allnames.index(key)].mean(0))
    fp = np.stack(rows)
    fp /= np.linalg.norm(fp, axis=1, keepdims=True)
    sub_gold = np.array([ITEM_LABEL[BLOCK_ITEMS[b]] for b in block])
    sidx = np.array([fine_names.index(s) for s in sub_gold])
    fam = np.array([s.split(":")[0] for s in fine_names])
    close = {}
    for tau in (0.01, 0.05, 0.1, 0.2):
        p = RouterParams(**{**BASE.__dict__, "tau": tau})
        pst, m = run(z, fp, p)
        famcov = np.array([(m[t] & (fam == gold[t])).any() for t in range(n)])
        close[f"hmm_tau={tau}"] = {**summarize(m, sidx, valid), "family_coverage": round(float(famcov[valid].mean()), 4)}
    for b in (1, 2, 4, 8):
        m = raw_topb(z, fp, BASE.window, b)
        famcov = np.array([(m[t] & (fam == gold[t])).any() for t in range(n)])
        close[f"raw_top{b}"] = {**summarize(m, sidx, valid), "family_coverage": round(float(famcov[valid].mean()), 4)}
    pst, m = run(z, fp, BASE)
    per_talk = {}
    for b in range(block.max() + 1):
        sel = (block == b) & valid
        per_talk[BLOCK_ITEMS[b]] = {"gold": ITEM_LABEL[BLOCK_ITEMS[b]], "coverage": round(float(m[sel][np.arange(sel.sum()), sidx[sel]].mean()), 4),
                                    "map_top1": fine_names[int(np.bincount(pst[sel, :-1].argmax(1), minlength=len(fine_names)).argmax())]}
    close["per_talk_hmm_base"] = per_talk
    res["close_domain"] = close
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

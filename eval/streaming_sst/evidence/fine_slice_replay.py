"""Open-loop routing over fine-grained slices that are close neighbours.

The NLP and medicine catalogs are split into subfield slices by llm_fine_slices.py,
and the eight other topics stay as single slices. On the saved translations of the
audited ten-talk run, each router ranks all slices for every decoder window, the
four highest-ranked slices form the working set, and a gold term occurrence counts
as covered when the slice that holds its term is in the working set of the window
in which it is spoken. Two routers are compared: cosine similarity between BGE-M3
embeddings of the context and of each slice description, and the language-model
letter scores of llm_router_replay.py.
"""

import argparse
import gzip
import json
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from llm_router_replay import LetterScorer, carry_forward, contexts_for
from router_replay import DESCRIPTIONS, EN, ZH, centroids

SR = 16000


def norm(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split())


def build_slices(fine):
    names, desc_llm, protos = [], {}, {}
    for domain, info in fine["domains"].items():
        for i, sub in enumerate(info["subfields"]):
            name = f"{domain}:{i}"
            names.append(name)
            desc_llm[name] = f"{sub['name']}: {sub['description_en']}"
            protos[name] = [sub["description_en"], sub["description_zh"]]
    for d in EN:
        if d not in fine["domains"]:
            names.append(f"{d}:all")
            desc_llm[f"{d}:all"] = f"{DESCRIPTIONS[d]} {EN[d]}"
            protos[f"{d}:all"] = [DESCRIPTIONS[d], EN[d], ZH[d]]
    return names, desc_llm, protos


def term_slices(fine):
    owner = {}
    for domain, info in fine["domains"].items():
        for a in info["assignment"]:
            owner.setdefault(norm(a["term"]), f"{domain}:{a['subfield']}")
    return owner


def coverage(scores, names, occ_window, occ_slice, k):
    order = np.argsort(-scores, axis=1)[:, :k]
    idx = {n: i for i, n in enumerate(names)}
    hit = [idx[s] in order[w] for w, s in zip(occ_window, occ_slice)]
    return float(np.mean(hit))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--fine", type=Path, required=True, help="llm_fine_slices.py output")
    ap.add_argument("--occurrences", type=Path, required=True, help="term_occurrences.jsonl.gz")
    ap.add_argument("--encoder", required=True, help="BGE-M3 path or id")
    ap.add_argument("--llm", required=True)
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fine = json.loads(args.fine.read_text(encoding="utf-8"))
    names, desc_llm, protos = build_slices(fine)
    owner = term_slices(fine)
    recs = json.load(open(args.run, encoding="utf-8"))["records"]
    starts = np.array([int(r["start_sample"]) / SR for r in recs])
    ends = np.array([int(r["cursor_samples"]) / SR for r in recs])
    texts = [str(r.get("text") or "") for r in recs]

    occ = [json.loads(line) for line in gzip.open(args.occurrences, "rt", encoding="utf-8")]
    occ_window, occ_slice, occ_domain, missing = [], [], [], []
    for o in occ:
        s = owner.get(norm(o["term"]))
        if s is None:
            missing.append(o["term"])
            continue
        w = int(np.searchsorted(ends, float(o["t_start"]), side="right"))
        occ_window.append(min(w, len(recs) - 1))
        occ_slice.append(s)
        occ_domain.append(o["domain"])
    occ_domain = np.array(occ_domain)

    ctx, nonempty = contexts_for(texts, args.w)
    result = {"slices": names, "occurrences": len(occ), "matched": len(occ_slice), "unmatched_terms": sorted(set(missing))[:50]}

    tok = AutoTokenizer.from_pretrained(args.encoder)
    tok.truncation_side = "left"
    enc = AutoModel.from_pretrained(args.encoder).to(device).eval()
    cnames, c = centroids(enc, tok, protos, device)
    assert cnames == names
    from router_replay import encode
    emb = encode(enc, tok, [x if x.strip() else " " for x in ctx], device)
    for t in range(1, len(emb)):
        if not nonempty[t]:
            emb[t] = emb[t - 1]
    cos = emb @ c.T
    del enc
    torch.cuda.empty_cache()

    scorer = LetterScorer(args.llm, device, args.batch)
    llm = carry_forward(scorer.score([x if x.strip() else "(none)" for x in ctx], names, desc_llm), nonempty)
    np.save(args.out.with_suffix(".cos.npy"), cos)
    np.save(args.out.with_suffix(".llm.npy"), llm)

    for label, scores in (("cosine", cos), ("llm", llm)):
        row = {}
        for k in (1, 2, 4, 6):
            row[f"term_cover@{k}"] = round(coverage(scores, names, occ_window, occ_slice, k), 4)
            for d in ("nlp", "medicine"):
                m = occ_domain == d
                row[f"term_cover@{k}_{d}"] = round(
                    coverage(scores, names, [w for w, keep in zip(occ_window, m) if keep], [s for s, keep in zip(occ_slice, m) if keep], k), 4
                )
        result[label] = row
        print(json.dumps({label: row}), flush=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

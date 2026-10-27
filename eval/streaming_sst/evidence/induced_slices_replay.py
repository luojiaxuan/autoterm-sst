"""Open-loop comparison of taxonomy slices and embedding-induced slices on one catalog.

Catalog: the ten-domain 100k catalog (ten taxonomy slices of 10k rows, cross-slice
duplicates kept). Items are its unique (term, Chinese) mappings. Partitions compared:
  taxonomy_prototype  the ten taxonomy slices, routed with the original three-line prototypes
  taxonomy_centroid   the same slices, routed with the mean embedding of their terms
  kmeans_K            spherical k-means on term embeddings (term; translation; description);
                      slice sizes follow the data
  random_K            a random partition with the k-means slice sizes
Centroids and k-means are fitted without benchmark-derived glossary sources; those items are
then assigned to the nearest centroid. For every decoder window of the audited run, slices
are admitted in order of cos(context, slice centroid) while the total stays within a term
budget B (larger slices that do not fit are skipped). The metric is the share of the 2,047
gold term occurrences whose term is inside the active slices at the window that completes it.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

from router_replay import VARIANTS, centroids, encode

DOMAINS = ("nlp", "medicine", "education", "finance", "legal", "environment", "entertainment", "science", "sports", "art")
BUDGETS = (2000, 5000, 10000, 20000, 40000)
KS = (10, 25, 50, 100, 200)
SEED = 20261026


def norm(term):
    return " ".join(str(term).lower().split())


def load_items(catalog_dir):
    items = {}
    for d in DOMAINS:
        for r in json.load(open(catalog_dir / f"{d}_core_10k.json", encoding="utf-8")):
            zh = str((r.get("target_translations") or {}).get("zh") or "")
            key = (norm(r["term"]), zh)
            item = items.setdefault(key, {"term": r["term"], "zh": zh, "desc": str(r.get("short_description") or ""), "sources": set(), "domains": set()})
            item["sources"].add(str(r.get("source")))
            item["domains"].add(d)
    return list(items.values())


def item_text(item):
    parts = [item["term"], item["zh"]] + ([item["desc"]] if item["desc"] else [])
    return "; ".join(parts)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def route(ctx_emb, windows, slice_centroids, slice_sizes, budget):
    order = np.argsort(-(ctx_emb[windows] @ slice_centroids.T), axis=1)
    smallest = int(slice_sizes.min())
    active = {}
    for w, row in zip(windows, order):
        used, chosen = 0, set()
        for s in row:
            if used + slice_sizes[s] <= budget:
                chosen.add(int(s))
                used += slice_sizes[s]
                if budget - used < smallest:
                    break
        active[w] = chosen
    return active


def coverage(active, labels_by_term, occ_windows, occ_terms):
    hits, present = [], []
    for w, term in zip(occ_windows, occ_terms):
        slices = labels_by_term.get(term)
        present.append(slices is not None)
        hits.append(slices is not None and bool(slices & active[w]))
    return np.array(hits), np.array(present)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-dir", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--occurrences", type=Path, required=True)
    ap.add_argument("--ctx-cache", type=Path, required=True, help="npz with key 'real' from hmm_router_replay.py")
    ap.add_argument("--term-cache", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exclude-source", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    items = load_items(args.catalog_dir)
    if args.term_cache.exists():
        term_emb = np.load(args.term_cache)["emb"]
    else:
        tok.truncation_side = "right"
        term_emb = encode(model, tok, [item_text(i) for i in items], device, batch=256)
        np.savez_compressed(args.term_cache, emb=term_emb)
    tok.truncation_side = "left"
    fit_mask = np.array([not (i["sources"] & set(args.exclude_source)) for i in items])

    recs = json.load(open(args.run, encoding="utf-8"))["records"]
    cursor = np.array([float(r["cursor_s"]) for r in recs])
    ctx_emb = np.load(args.ctx_cache)["real"]
    occ = [json.loads(line) for line in gzip.open(args.occurrences, "rt", encoding="utf-8")]
    occ_windows = [min(int(np.searchsorted(cursor, o["t_end"])), len(recs) - 1) for o in occ]
    occ_terms = [norm(o["term"]) for o in occ]
    occ_domain = np.array([o["domain"] for o in occ])
    windows = sorted(set(occ_windows))

    def evaluate(labels, slice_centroids, extra=None):
        sizes = np.bincount(labels, minlength=len(slice_centroids)) if labels.ndim == 1 else labels.sum(0)
        if labels.ndim == 1:
            by_term = {}
            for i, lab in zip(items, labels):
                by_term.setdefault(norm(i["term"]), set()).add(int(lab))
        else:
            by_term = {}
            for i, row in zip(items, labels):
                by_term.setdefault(norm(i["term"]), set()).update(int(s) for s in np.flatnonzero(row))
        row = {"slices": len(slice_centroids), "size_min": int(sizes.min()), "size_median": float(np.median(sizes)), "size_max": int(sizes.max()), **(extra or {})}
        for budget in BUDGETS:
            active = route(ctx_emb, windows, slice_centroids, sizes, budget)
            hits, present = coverage(active, by_term, occ_windows, occ_terms)
            row[f"B={budget}"] = {
                "cover": round(float(hits.mean()), 4),
                "cover_nlp": round(float(hits[occ_domain == "nlp"].mean()), 4),
                "cover_medicine": round(float(hits[occ_domain == "medicine"].mean()), 4),
                "mean_active_slices": round(float(np.mean([len(a) for a in active.values()])), 2),
            }
        row["catalog_ceiling"] = round(float(present.mean()), 4)
        return row

    result = {
        "catalog": str(args.catalog_dir),
        "items": len(items),
        "fit_items": int(fit_mask.sum()),
        "excluded_sources": args.exclude_source,
        "occurrences": len(occ),
        "budgets": BUDGETS,
        "partitions": {},
    }
    tax = np.array([[d in i["domains"] for d in DOMAINS] for i in items])
    names, proto = centroids(model, tok, VARIANTS["original"], device)
    proto = proto[[names.index(d) for d in DOMAINS]]
    result["partitions"]["taxonomy_prototype"] = evaluate(tax, proto)
    tax_centroids = unit(np.stack([term_emb[tax[:, j] & fit_mask].mean(0) for j in range(len(DOMAINS))]))
    result["partitions"]["taxonomy_centroid"] = evaluate(tax, tax_centroids)

    fit_emb = unit(term_emb[fit_mask])
    rng = np.random.default_rng(SEED)
    for k in KS:
        km = KMeans(n_clusters=k, n_init=3, random_state=SEED).fit(fit_emb)
        cent = unit(km.cluster_centers_)
        labels = np.argmax(unit(term_emb) @ cent.T, axis=1)
        result["partitions"][f"kmeans_{k}"] = evaluate(labels, cent)
        random_labels = rng.permutation(labels)
        random_cent = unit(np.stack([term_emb[(random_labels == j) & fit_mask].mean(0) for j in range(k)]))
        result["partitions"][f"random_{k}"] = evaluate(random_labels, random_cent)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

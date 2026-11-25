"""Build the routed, merged and known-domain indexes of one catalog scale from one text encoding.

100k: the ten 10k topic slices. 1m: the same ten slices grown to about 101k pairs each with general pairs.
260k: the ten 10k slices plus one 10k slice for each topic of --topics, filled with the first general pairs
under a balanced assignment among the added topics only, so the ten original slices stay exactly as in 100k.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import sys
from pathlib import Path

import torch

from catalog import conflicts, distinct, extend, pair_of

PREFIX = "{}_core_10k"
MERGED = "merged_all"


def load(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def prototypes(meta: dict, domains: list) -> dict:
    """The texts the router reads: the slice description, then the domain's prototype sentences."""
    from framework.agents.term_memory.domain_taxonomy import DOMAIN_ROUTER_PROTOTYPES

    return {
        d: tuple(dict.fromkeys([str(meta[f"{d}_core_1k"]["description"]).strip(), *DOMAIN_ROUTER_PROTOTYPES[d]]))
        for d in domains
    }


def balance(sim: torch.Tensor, target: float, tolerance: float = 0.005) -> torch.Tensor:
    """Each pair's topic under additive topic prices that give every topic ``target`` pairs.

    Raising the price of an oversubscribed topic moves its least similar pairs to their next best
    topic, so the result approximates the equal-size assignment of maximum total similarity.
    """
    price = torch.zeros(sim.shape[1], device=sim.device)
    rate = 0.01
    best = (float("inf"), None)
    for step in range(1500):
        pick = (sim - price).argmax(1)
        count = torch.bincount(pick, minlength=sim.shape[1]).float()
        off = float((count / target - 1).abs().max())
        if off < best[0]:
            best = (off, pick)
        if off < tolerance:
            break
        price += rate * (count / target).clamp(min=0.05).log()
        if step % 150 == 149:
            rate *= 0.7
    if best[0] > 2 * tolerance:
        raise ValueError(f"no balanced assignment within {2 * tolerance:.1%}: {best[0]:.2%}")
    return best[1]


def assign(added: list, texts: dict, model: str, device: str, layout: str) -> list:
    """Topic of every added pair by prototype similarity of "term, description": nearest, or balanced."""
    from framework.agents.term_memory.context_similarity import DomainDescriptionSimilarity

    scorer = DomainDescriptionSimilarity(model_id=model, device=device, batch_size=512, prototypes=texts)
    scorer._load()
    rows = []
    for start in range(0, len(added), 8192):
        part = [f"{e['term']}, {e['desc']}" for e in added[start : start + 8192]]
        rows.append(scorer._encode_sync(part) @ scorer._centroids.float().T)
    sim = torch.cat(rows)
    pick = sim.argmax(1) if layout == "topical" else balance(sim, len(added) / sim.shape[1])
    return [scorer._domains[i] for i in pick.tolist()]


def encode(terms: list, retriever_code: Path, checkpoint: Path, device: str, text_model: str | None) -> torch.Tensor:
    sys.path.insert(0, str(retriever_code))
    import build_maxsim_index as builder

    if text_model:
        # A local snapshot keeps the tokenizer from querying the Hub, which fails in offline mode.
        builder.TEXT_MODEL_ID = text_model

    dev = torch.device(device)
    encoder, tokenizer = builder.build_text_encoder(dev, builder.TEXT_LORA_RANK, builder.TEXT_LORA_ALPHA)
    saved = torch.load(checkpoint, map_location="cpu")
    if "text_model_state_dict" not in saved:
        raise ValueError("the retriever checkpoint has no text encoder")
    del saved
    builder.load_text_checkpoint(encoder, str(checkpoint), dev)
    parts = []
    for start in range(0, len(terms), 65536):
        part = [{"term": t} for t in terms[start : start + 65536]]
        parts.append(builder.encode_terms(part, encoder, tokenizer, dev).cpu())
    return torch.cat(parts)


def stored(terms: list, sources: list) -> torch.Tensor:
    """Vectors of the terms from indexes the same retriever built on GPU; the first source holding a term wins.

    Re-encoding in another software environment does not reproduce those vectors, and the published
    indexes were built from them.
    """
    rows_of = {}
    for i, term in enumerate(terms):
        rows_of.setdefault(term.strip(), []).append(i)
    vectors, filled = None, torch.zeros(len(terms), dtype=torch.bool)
    for source in sources:
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if vectors is None:
            vectors = torch.empty((len(terms), saved["text_embs"].shape[1]), dtype=saved["text_embs"].dtype)
        rows, cols = [], []
        for j, entry in enumerate(saved["term_list"]):
            for i in rows_of.get(str(entry.get("term", "")).strip(), ()):
                if not filled[i]:
                    rows.append(i)
                    cols.append(j)
                    filled[i] = True
        if rows:
            vectors[rows] = saved["text_embs"][cols].to(vectors.dtype)
        print(f"vectors from {source}: {len(rows)}", flush=True)
        del saved
    if not bool(filled.all()):
        raise ValueError(f"{int((~filled).sum())} terms have no stored vector")
    return vectors


def check_encoding(vectors: torch.Tensor, row: dict, reference: Path) -> float:
    """Lowest cosine between this encoding and the published 1k-slice indexes."""
    lowest = 1.0
    paths = sorted(reference.rglob("*_core_1k/en-zh/maxsim.pt"))
    if len(paths) != 10:
        raise ValueError(f"expected ten published indexes, found {len(paths)}")
    for path in paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        known = [i for i, e in enumerate(saved["term_list"]) if e["term"].strip() in row]
        new = vectors[[row[saved["term_list"][i]["term"].strip()] for i in known]].float()
        cos = torch.nn.functional.cosine_similarity(new, saved["text_embs"][known].float())
        lowest = min(lowest, float(cos.min()))
    return lowest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--retriever-code", type=Path, required=True)
    ap.add_argument("--retriever", type=Path, required=True)
    ap.add_argument("--context-model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", choices=("100k", "260k", "1m"), required=True)
    ap.add_argument("--topics", type=Path,
                    help="260k only: added topics as {domain: {description, prototypes}}")
    ap.add_argument("--slice-size", type=int, default=10_000, help="260k only: pairs per added topic")
    ap.add_argument("--layout", choices=("balanced", "topical"), default="balanced",
                    help="1m only: equal-size slices, or each pair in its nearest topic")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text-model", help="local snapshot of the retriever's base text model (BAAI/bge-m3)")
    ap.add_argument("--vector-sources", type=Path, nargs="+",
                    help="GPU-built indexes of the same retriever to take every term's vector from instead of encoding")
    args = ap.parse_args()

    template = json.loads((args.reference / "manifest.json").read_text(encoding="utf-8"))
    slices = load(args.bundle / "slices.json.gz")
    domains = list(slices)
    texts = prototypes(template["preset_meta"], domains)
    topics = {}
    if args.scale == "1m":
        added = load(args.bundle / "added.json.gz")
        slices = extend(slices, added, assign(added, texts, args.context_model, args.device, args.layout))
        torch.cuda.empty_cache()
    elif args.scale == "260k":
        topics = json.loads(args.topics.read_text(encoding="utf-8"))
        if set(topics) & set(domains):
            raise ValueError("an added topic repeats one of the ten slices")
        new = {d: tuple(dict.fromkeys([topics[d]["description"], *topics[d]["prototypes"]])) for d in topics}
        added = load(args.bundle / "added.json.gz")[: args.slice_size * len(new)]
        slices = extend({**slices, **{d: [] for d in new}}, added,
                        assign(added, new, args.context_model, args.device, "balanced"))
        torch.cuda.empty_cache()
        domains += list(new)
        texts.update(new)
    merged = distinct(e for d in domains for e in slices[d])
    if {pair_of(e) for d in domains for e in slices[d]} != {pair_of(e) for e in merged}:
        raise ValueError("the slices and the merged index differ")

    terms = list(dict.fromkeys(e["term"] for d in domains for e in slices[d]))
    row = {t: i for i, t in enumerate(terms)}
    if args.vector_sources:
        vectors = stored(terms, args.vector_sources)
    else:
        vectors = encode(terms, args.retriever_code, args.retriever, args.device, args.text_model)
    lowest = check_encoding(vectors, row, args.reference)
    if lowest < 0.999:
        raise ValueError(f"the encoding differs from the published indexes: cosine {lowest:.5f}")

    manifest = {k: copy.deepcopy(v) for k, v in template.items() if k not in ("scales", "preset_meta")}
    manifest.update(root=str(args.out), snapshot_id=f"run_{args.scale}", scales={}, preset_meta={})
    named = [(PREFIX.format(d), d, slices[d]) for d in domains] + [(MERGED, None, merged)]
    for preset, domain, entries in named:
        folder = args.out / preset
        (folder / "en-zh").mkdir(parents=True, exist_ok=True)
        glossary = [{"term": e["term"], "target_translations": {"zh": e["zh"]}} for e in entries]
        (folder / "glossary.json").write_text(json.dumps(glossary, ensure_ascii=False), encoding="utf-8")
        listed = [{"key": e["term"].lower(), **e} for e in glossary]
        index = folder / "en-zh" / "maxsim.pt"
        torch.save({"text_embs": vectors[[row[e["term"]] for e in entries]].clone(), "term_list": listed}, index)
        paths = {"terms_path": str(folder / "glossary.json"), "indexes": {"maxsim": str(index)}}
        manifest["scales"][preset] = {"en-zh": {**paths, "num_terms": len(entries)}}

        source = f"{domain}_core_1k" if domain else "merged_10x1k"
        if domain in topics:
            # An added topic takes the field layout of an original slice with its own domain and description.
            meta = copy.deepcopy(template["preset_meta"]["art_core_1k"])
            meta.update(domain=domain, domain_id=domain, description=topics[domain]["description"])
        else:
            meta = copy.deepcopy(template["preset_meta"].get(source, {}))
        for key in ("glossary_sha256", "maxsim_index_sha256", "source_term_fingerprint_sha256", "prefix_size"):
            meta.pop(key, None)
        meta.update(
            id=preset, preset_id=preset, label=preset, parent_preset_id="",
            glossary_path=paths["terms_path"], maxsim_index_path=str(index), term_count=len(entries),
        )
        manifest["preset_meta"][preset] = meta
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # The router reads these texts: the server's default for the ten original slices, and the added topics'
    # descriptions and prototypes, which the framework does not know.
    (args.out / "prototypes.json").write_text(json.dumps({d: list(texts[d]) for d in domains}, ensure_ascii=False,
                                                         indent=1), encoding="utf-8")

    report = {
        "scale": args.scale,
        "layout": {"1m": args.layout, "100k": "ten 10k slices",
                   "260k": f"ten 10k slices and {len(topics)} balanced {args.slice_size} slices"}[args.scale],
        "merged_pairs": len(merged),
        "slice_pairs": {d: len(slices[d]) for d in domains},
        "terms_with_conflicting_translations": {d: conflicts(slices[d]) for d in domains},
        "merged_terms_with_conflicting_translations": conflicts(merged),
        "budget": sum(len(slices[d]) for d in domains),
        "lowest_cosine_to_published_index": lowest,
        "prototypes": texts,
    }
    (args.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()

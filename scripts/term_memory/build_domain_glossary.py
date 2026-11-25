#!/usr/bin/env python3
"""Cut a relevant, domain-ranked glossary from the large Wikidata glossary.

Generic Wikidata is dominated by people/places/taxa, which makes a uniform slice
name-heavy and a poor demo. This builds a *relevant* domain slice instead:

* seed with a curated, already-translated domain list (e.g.
  ``wiki_glossary_nlp_ai_cs_enriched.json``) so the headline terms are present;
* expand by streaming the big translated glossary
  (``glossary_filtered_from_wiki.json``, ~12.4M entries) and keeping entries
  whose term / short_description match the domain's keywords, have a target-lang
  translation, and pass the shared quality filter (``filter_terms.keep_term``);
* de-dup (seed first), cap to ``--limit``, emit a list-shaped glossary JSON ready
  for ``build_maxsim_index.py`` + ``build_term_memory_snapshot.py``.

    python scripts/term_memory/build_domain_glossary.py --domain academic \
        --glossary <glossary_filtered_from_wiki.json> \
        --seed <wiki_glossary_nlp_ai_cs_enriched.json> \
        --target-lang zh --limit 100000 --out wiki_academic_zh_100k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.term_memory.filter_terms import keep_term  # noqa: E402

# Domain keyword sets matched against lowercased "term + short_description".
DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "academic": (
        "algorithm", "machine learning", "deep learning", "neural network", "neural net",
        "natural language", "language model", "linguistic", "computational", "computer science",
        "artificial intelligence", "dataset", "data set", "statistical", "probability",
        "optimization", "semantic", "syntax", "speech recognition", "speech translation",
        "machine translation", "embedding", "transformer", "corpus", "information retrieval",
        "reinforcement learning", "supervised", "unsupervised", "classification", "regression",
        "clustering", "gradient", "theorem", "mathematic", "physics", "chemistry", "quantum",
        "graph theory", "entropy", "bayesian", "markov", "convolution", "attention mechanism",
        "tokeniz", "parsing", "named entity", "sentiment analysis", "question answering",
        "summarization", "knowledge graph", "ontology", "inference", "encoder", "decoder",
        "benchmark", "metric", "annotation", "treebank", "word embedding", "fine-tuning",
        "pretrain", "self-supervised", "diffusion model", "generative model", "scientific",
    ),
    "medicine": (
        "disease", "syndrome", "disorder", "infection", "cancer", "tumor", "tumour", "therapy",
        "treatment", "drug", "medication", "pharmacolog", "clinical", "diagnosis", "symptom",
        "pathogen", "virus", "bacteri", "antibiotic", "vaccine", "surgery", "surgical",
        "anatomy", "anatomical", "physiolog", "gene ", "protein", "enzyme", "hormone",
        "receptor", "immun", "neurolog", "cardio", "oncolog", "psychiatr", "epidemi", "medical",
    ),
}

# Types that make the "general" slice noisy; excluded there (academic/medicine
# use keyword inclusion instead, so these don't apply to them).
_GENERAL_DROP = ("name", "wikimedia", "given name", "family name")


def domain_match(domain: str, term: str, desc: str) -> bool:
    blob = (term + " " + desc).lower()
    kws = DOMAIN_KEYWORDS.get(domain)
    if kws is None:  # "general": keep anything that isn't obviously a bare name
        return not any(d in desc.lower() for d in _GENERAL_DROP)
    return any(kw in blob for kw in kws)


def _entry_row(term: str, entry: Dict[str, Any], key: str = "") -> Dict[str, Any]:
    tt = entry.get("target_translations") if isinstance(entry, dict) else None
    return {
        "term": term,
        "term_key": (entry.get("term_key") if isinstance(entry, dict) else None) or key or term.lower(),
        "target_translations": tt if isinstance(tt, dict) else {},
        "description": str((entry.get("short_desc") or entry.get("short_description")
                            or entry.get("description") or "") if isinstance(entry, dict) else ""),
        "source": "wikidata",
    }


def _iter_glossary(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    try:
        import ijson  # type: ignore
        with path.open("rb") as fh:
            head = fh.read(1)
        with path.open("rb") as fh:
            if head == b"{":
                for key, entry in ijson.kvitems(fh, ""):
                    yield str(key), entry
            else:
                for entry in ijson.items(fh, "item"):
                    yield "", entry
        return
    except ImportError:
        pass
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.items() if isinstance(data, dict) else ((("", e) for e in data))
    for key, entry in items:
        yield str(key), entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glossary", required=True, help="big translated glossary (glossary_filtered_from_wiki.json)")
    ap.add_argument("--seed", action="append", default=[], help="curated translated glossary(s) to prioritize")
    ap.add_argument("--domain", default="academic", help="academic | medicine | general")
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    code = args.target_lang.strip().lower()
    seen = set()
    kept: List[Dict[str, Any]] = []

    def consider(term: str, entry: Dict[str, Any], key: str = "", *, require_domain: bool) -> None:
        if len(kept) >= args.limit:
            return
        row = _entry_row(term, entry, key)
        if code not in (row["target_translations"] or {}):
            return
        ok, _ = keep_term(row, code)
        if not ok:
            return
        if require_domain and not domain_match(args.domain, row["term"], row["description"]):
            return
        k = row["term_key"].lower()
        if k in seen:
            return
        seen.add(k)
        kept.append(row)

    # 1) curated seed(s) first (relevance guaranteed)
    for seed_path in args.seed:
        sp = Path(os.path.expandvars(seed_path))
        if not sp.is_file():
            print(f"[domain] WARN seed missing: {sp}")
            continue
        data = json.loads(sp.read_text(encoding="utf-8"))
        vals = data if isinstance(data, list) else list(data.values())
        before = len(kept)
        for entry in vals:
            if isinstance(entry, dict):
                consider(str(entry.get("term") or ""), entry, require_domain=False)
        print(f"[domain] seed {sp.name}: +{len(kept)-before} (total {len(kept)})")

    # 2) expand from the big glossary by domain keywords
    seen_g = 0
    for key, entry in _iter_glossary(Path(os.path.expandvars(args.glossary))):
        seen_g += 1
        if isinstance(entry, dict):
            consider(str(entry.get("term") or key), entry, key, require_domain=True)
        if seen_g % 1000000 == 0:
            print(f"[domain] scanned {seen_g} glossary entries, kept {len(kept)}", flush=True)
        if len(kept) >= args.limit:
            print(f"[domain] hit limit {args.limit} after {seen_g} entries")
            break

    out = Path(os.path.expandvars(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    glossary = [
        {
            "term": r["term"],
            "term_key": r["term_key"],
            "target_translations": r["target_translations"],
            "short_description": r["description"],
            "source": r["source"],
        }
        for r in kept
    ]
    out.write_text(json.dumps(glossary, ensure_ascii=False), encoding="utf-8")
    print(f"[domain] wrote {len(glossary)} '{args.domain}' terms (target={code}) -> {out}")


if __name__ == "__main__":
    main()

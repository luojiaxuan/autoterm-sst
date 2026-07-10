#!/usr/bin/env python3
"""Build ten exact-size AutoTerm slices from auditable per-domain sources.

NLP and medicine may use the evaluated project glossaries as seeds.  The other
RealSI domains must be supplied as pre-collected RDF/category candidates; this
builder intentionally does not infer a domain from term substrings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.domain_taxonomy import DOMAIN_TO_PRESET, WORKING_DOMAINS
from scripts.term_memory.filter_terms import keep_term


def normalized_term(entry: Mapping[str, Any]) -> str:
    return " ".join(str(entry.get("term") or "").strip().casefold().split())


def translation_of(entry: Mapping[str, Any], target_lang: str) -> str:
    translations = entry.get("target_translations")
    if isinstance(translations, Mapping):
        return str(translations.get(target_lang) or "").strip()
    return ""


def load_glossary(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = list(payload.values()) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise ValueError(f"{path}: expected a list/dict of glossary objects")
    return [dict(item) for item in values]


def parse_domain_paths(values: Sequence[str], domains: Sequence[str], flag: str) -> Dict[str, List[Path]]:
    result = {domain: [] for domain in domains}
    for value in values:
        domain, separator, raw_path = value.partition("=")
        domain = domain.strip()
        if not separator or domain not in result or not raw_path.strip():
            raise ValueError(f"invalid {flag} {value!r}; expected domain=/path/to/glossary.json")
        result[domain].append(Path(raw_path).expanduser())
    return result


def _has_domain_provenance(entry: Mapping[str, Any]) -> bool:
    source = str(entry.get("source") or "")
    if source == "wikidata_p31_p279":
        return bool(entry.get("wikidata_qid") and entry.get("domain_root_qid") and entry.get("rdf_path"))
    if source == "wikidata_exact_p31":
        return bool(entry.get("wikidata_qid") and entry.get("wikidata_type_qid") and entry.get("rdf_path") == "P31")
    if source == "wikipedia_category":
        path = entry.get("category_path")
        return bool(entry.get("wikipedia_pageid") and entry.get("wikidata_qid") and isinstance(path, list) and path)
    if source == "wikipedia_deep_category":
        path = entry.get("category_path")
        return bool(entry.get("wikipedia_pageid") and entry.get("wikidata_qid") and isinstance(path, list) and path)
    return False


def _quality_entry(
    entry: Mapping[str, Any],
    target_lang: str,
    *,
    require_domain_provenance: bool = False,
) -> Dict[str, Any] | None:
    if not translation_of(entry, target_lang):
        return None
    if require_domain_provenance and not _has_domain_provenance(entry):
        return None
    row = dict(entry)
    row.setdefault("term_key", normalized_term(row))
    probe = {
        "term": row.get("term"),
        "target_translations": row.get("target_translations"),
        "description": row.get("short_description") or row.get("description") or "",
    }
    ok, _ = keep_term(probe, target_lang)
    return row if ok else None


def build_catalog(
    *,
    domains: Sequence[str],
    seeds: Mapping[str, Sequence[Path]],
    domain_sources: Mapping[str, Sequence[Path]],
    target_lang: str,
    limit: int,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    rows: Dict[str, List[Dict[str, Any]]] = {domain: [] for domain in domains}
    seen_by_domain: Dict[str, set[str]] = {domain: set() for domain in domains}
    roles: Dict[str, Counter[str]] = {domain: Counter() for domain in domains}

    for domain in domains:
        for path in seeds.get(domain, ()):
            for raw in load_glossary(path):
                key = normalized_term(raw)
                if (
                    not key
                    or not translation_of(raw, target_lang)
                    or len(rows[domain]) >= limit
                ):
                    continue
                seen_by_domain[domain].add(key)
                rows[domain].append(dict(raw))
                roles[domain]["seed"] += 1

    global_seen = {key for domain_seen in seen_by_domain.values() for key in domain_seen}
    for domain in domains:
        for path in domain_sources.get(domain, ()):
            for raw in load_glossary(path):
                if len(rows[domain]) >= limit:
                    break
                entry = _quality_entry(raw, target_lang, require_domain_provenance=True)
                key = normalized_term(raw)
                if entry is None or not key or key in global_seen:
                    continue
                global_seen.add(key)
                seen_by_domain[domain].add(key)
                rows[domain].append(entry)
                roles[domain][str(entry.get("source") or "domain_source")] += 1

    underfilled = {domain: len(entries) for domain, entries in rows.items() if len(entries) != limit}
    if underfilled:
        missing_sources = [
            domain
            for domain, count in underfilled.items()
            if not seeds.get(domain) and not domain_sources.get(domain)
        ]
        suffix = f"; no source supplied for {missing_sources}" if missing_sources else ""
        raise RuntimeError(f"could not build exact {limit}-row slices: {underfilled}{suffix}")

    merged = [entry for domain in domains for entry in rows[domain]]
    unique_terms = len({normalized_term(entry) for entry in merged})
    unique_mappings = len(
        {
            (normalized_term(entry), translation_of(entry, target_lang).casefold())
            for entry in merged
        }
    )
    report = {
        "construction": "seeded evaluated slices plus Wikidata P31/P279 and Wikipedia category paths",
        "domain_inference_from_substrings": False,
        "target_lang": target_lang,
        "domains": list(domains),
        "slice_size": limit,
        "merged_rows": len(merged),
        "merged_unique_terms": unique_terms,
        "merged_unique_mappings": unique_mappings,
        "cross_slice_duplicate_terms": len(merged) - unique_terms,
        "per_domain": {
            domain: {
                "preset": DOMAIN_TO_PRESET[domain],
                "rows": len(rows[domain]),
                "seed_paths": [str(path) for path in seeds.get(domain, ())],
                "domain_source_paths": [str(path) for path in domain_sources.get(domain, ())],
                "source_roles": dict(roles[domain]),
            }
            for domain in domains
        },
    }
    return rows, report


def write_catalog(
    rows: Mapping[str, Sequence[Dict[str, Any]]],
    report: Dict[str, Any],
    *,
    out_dir: Path,
    target_lang: str,
    limit: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for domain, entries in rows.items():
        path = out_dir / f"{DOMAIN_TO_PRESET[domain]}.json"
        path.write_text(json.dumps(list(entries), ensure_ascii=False), encoding="utf-8")
        paths[domain] = str(path)

    merged = [entry for domain in rows for entry in rows[domain]]
    merged_path = out_dir / f"merged_realsi_{len(merged) // 1000}k_{target_lang}.json"
    merged_path.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    report.update(
        {
            "slice_paths": paths,
            "merged_path": str(merged_path),
            "expected_merged_rows": limit * len(rows),
        }
    )
    report_path = out_dir / "catalog_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    ap.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Repeat domain=/path; evaluated seed rows are prioritized.",
    )
    ap.add_argument(
        "--domain-source",
        action="append",
        default=[],
        help="Repeat domain=/path; rows must carry RDF/category provenance.",
    )
    ap.add_argument("--source-revision", default="")
    args = ap.parse_args()

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    unknown = [item for item in domains if item not in DOMAIN_TO_PRESET]
    if unknown:
        raise SystemExit(f"unknown domains: {', '.join(unknown)}")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    try:
        seeds = parse_domain_paths(args.seed, domains, "--seed")
        sources = parse_domain_paths(args.domain_source, domains, "--domain-source")
        rows, report = build_catalog(
            domains=domains,
            seeds=seeds,
            domain_sources=sources,
            target_lang=args.target_lang,
            limit=args.limit,
        )
        if args.source_revision:
            report["source_revision"] = args.source_revision
        report = write_catalog(
            rows,
            report,
            out_dir=args.out_dir.expanduser(),
            target_lang=args.target_lang,
            limit=args.limit,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

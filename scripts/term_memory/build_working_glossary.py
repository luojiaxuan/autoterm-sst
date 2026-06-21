#!/usr/bin/env python3
"""Build compact working-glossary slices from translated open-memory rows.

This is the CPU-side half of the zero-setup adaptive glossary pipeline. It
filters/ranks an existing translated glossary into 5k-10k runtime slices such as
``common_10k`` and ``nlp_core_10k``. MaxSim index construction still happens via
the RASST GPU builder; this script records the expected index path in the
manifest and can publish before or after the index exists.

Artifacts are written under ``$RASST_DEMO_DATA_ROOT/runtime/term_memory`` by
default and should not be committed to git.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.agents.term_memory import TermEntry, lang_key  # noqa: E402
from framework.agents.term_memory.domain_taxonomy import (  # noqa: E402
    DOMAIN_TO_PRESET,
    GENERAL_DOMAIN,
    WORKING_PRESET_META,
    entry_domain_score,
)
from scripts.term_memory.filter_terms import keep_term  # noqa: E402
from scripts.term_memory.publish_manifest import publish  # noqa: E402


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    yield row
        return
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                row = dict(value)
                row.setdefault("term", key)
                yield row
    elif isinstance(raw, list):
        for value in raw:
            if isinstance(value, dict):
                yield value


def _translation(row: Dict[str, Any], target_lang: str) -> str:
    translations = row.get("target_translations")
    if isinstance(translations, dict):
        return str(translations.get(target_lang) or "").strip()
    if str(row.get("target_lang") or "").lower() in {"", target_lang}:
        return str(row.get("target_label") or row.get("translation") or "").strip()
    return ""


def _term(row: Dict[str, Any]) -> str:
    return str(row.get("term") or row.get("source_label") or "").strip()


def _normalize_row(row: Dict[str, Any], target_lang: str) -> Dict[str, Any]:
    term = _term(row)
    translation = _translation(row, target_lang)
    out = dict(row)
    out["term"] = term
    out["term_key"] = str(row.get("term_key") or term.lower())
    out["target_translations"] = {target_lang: translation}
    out["source"] = str(row.get("source") or "wikidata")
    if row.get("short_desc") and not row.get("short_description"):
        out["short_description"] = row.get("short_desc")
    return out


def _slice_domain(slice_name: str) -> str:
    for domain, preset in DOMAIN_TO_PRESET.items():
        if preset == slice_name:
            return domain
    return GENERAL_DOMAIN


def _build_slice(
    rows: List[Dict[str, Any]],
    *,
    slice_name: str,
    target_lang: str,
    limit: int,
) -> List[Dict[str, Any]]:
    domain = _slice_domain(slice_name)
    seen = set()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for raw in rows:
        row = _normalize_row(raw, target_lang)
        if not row["term"] or not row["target_translations"].get(target_lang):
            continue
        ok, _ = keep_term(row, target_lang)
        if not ok:
            continue
        key = row["term_key"].lower()
        if key in seen:
            continue
        seen.add(key)
        score = entry_domain_score(row, domain)
        if domain != GENERAL_DOMAIN and score <= 0.0:
            continue
        scored.append((score, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in scored[:limit]]


def _write_glossary(path: Path, rows: List[Dict[str, Any]], target_lang: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "term": row["term"],
            "term_key": row.get("term_key") or row["term"].lower(),
            "target_translations": row["target_translations"],
            "short_description": row.get("short_description") or row.get("description") or "",
            "source": row.get("source") or "wikidata",
        }
        for row in rows
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_terms(path: Path, rows: List[Dict[str, Any]], target_lang: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            entry = TermEntry(
                term_id=str(row.get("term_id") or row.get("qid") or ""),
                source_lang="en",
                target_lang=target_lang,
                source_label=row["term"],
                target_label=row["target_translations"][target_lang],
                entity_types=list(row.get("entity_types") or []),
                domains=[_slice_domain(str(row.get("_slice") or ""))] if row.get("_slice") else [],
                source=str(row.get("source") or "wikidata"),
                source_url=str(row.get("source_url") or ""),
                updated_at=str(row.get("updated_at") or ""),
            )
            handle.write(entry.to_jsonl_line() + "\n")
    return len(rows)


def _parse_indexes(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"expected <slice>=<path>, got: {value}")
        key, path = value.split("=", 1)
        out[key.strip()] = os.path.expandvars(path.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="translated glossary JSON or normalized JSONL rows")
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--slices", default="common_10k,nlp_core_10k,medicine_core_10k,finance_core_10k,legal_core_10k")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument(
        "--root",
        default=os.environ.get(
            "RASST_DEMO_TERM_MEMORY_ROOT",
            "/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory",
        ),
    )
    ap.add_argument("--snapshot-id", default="")
    ap.add_argument("--source", default="wikidata/wikipedia-derived")
    ap.add_argument("--maxsim-index", action="append", default=[], help="<slice>=<existing maxsim .pt>")
    ap.add_argument("--require-index", action="store_true", help="refuse to publish if any index path is missing")
    args = ap.parse_args()

    target_lang = args.target_lang.strip().lower()
    slices = [s.strip() for s in args.slices.split(",") if s.strip()]
    root = Path(os.path.expandvars(args.root))
    snapshot_id = args.snapshot_id or (
        "working_" + _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    rows = list(_iter_rows(Path(os.path.expandvars(args.input))))
    indexes = _parse_indexes(args.maxsim_index)
    print(f"[working] loaded {len(rows)} rows from {args.input}")

    scales: Dict[str, Dict[str, Dict[str, Any]]] = {}
    preset_meta: Dict[str, Dict[str, Any]] = {}
    for slice_name in slices:
        kept = _build_slice(rows, slice_name=slice_name, target_lang=target_lang, limit=args.limit)
        for row in kept:
            row["_slice"] = slice_name
        glossary_path = root / "glossaries" / f"{slice_name}.{target_lang}.json"
        terms_path = root / "snapshots" / snapshot_id / f"{slice_name}.en-{target_lang}.jsonl"
        index_path = indexes.get(slice_name) or str(root / "indexes" / slice_name / f"en-{target_lang}" / "maxsim.pt")
        _write_glossary(glossary_path, kept, target_lang)
        n = _write_terms(terms_path, kept, target_lang)
        key = lang_key(target_lang)
        scales[slice_name] = {
            key: {
                "terms_path": str(terms_path),
                "indexes": {"maxsim": index_path},
                "num_terms": n,
            }
        }
        meta = dict(WORKING_PRESET_META.get(slice_name) or {})
        domain_id = str(meta.get("domain") or "").strip() or "general"
        meta.update(
            {
                "id": slice_name,
                "preset_id": slice_name,
                "domain_id": domain_id,
                "fallback_preset_id": "common_10k" if slice_name != "common_10k" else "",
                "enabled_for_auto_router": True,
                "terms": n,
                "term_count": n,
                "glossary_path": str(glossary_path),
                "index_path": index_path,
                "maxsim_index_path": index_path,
                "centroid_path": str(root / "centroids" / f"{slice_name}.pt"),
                "source": args.source,
                "snapshot_id": snapshot_id,
            }
        )
        preset_meta[slice_name] = meta
        print(f"[working] {slice_name}: {n} terms -> {glossary_path}")
        print(f"[working] {slice_name}: expected maxsim -> {index_path}")

    manifest = {
        "snapshot_id": snapshot_id,
        "source": args.source,
        "created_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
        "root": str(root),
        "scales": scales,
        "preset_meta": preset_meta,
    }
    archived, current = publish(manifest, root / "manifests", require_index=args.require_index)
    print(f"[working] published manifest: {archived}")
    print(f"[working] current manifest : {current}")


if __name__ == "__main__":
    main()

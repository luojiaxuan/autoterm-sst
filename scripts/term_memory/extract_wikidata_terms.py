#!/usr/bin/env python3
"""Normalize Wikidata-derived glossary files into open-memory term rows.

The upstream RDF extraction (``extract_rdf_terms_with_p31.py`` over
``latest-truthy.nt``) and translation join already produced, on disk:

* ``glossary_filtered_from_wiki.json`` — dict ``{term_key: {term,
  target_translations:{zh,de,...}, short_description}}`` (~12.4M entries);
* P31-ranked + sampled scale files ``wiki_p31_..._sampleNNN_zh.json`` — lists of
  ``{term, term_key, rank, source, target_translations, short_description}``.

This script reads either shape and emits one normalized row per line::

    {"term","term_key","target_translations","description","source","rank","entity_types"}

so ``filter_terms.py`` then ``build_term_memory_snapshot.py`` can consume a single
stable format. The huge dict is streamed with ``ijson`` when available (falls
back to ``json.load`` for the smaller list/sample files).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def normalize_entry(key: str, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """One glossary entry -> a normalized row (or None if unusable)."""

    if not isinstance(entry, dict):
        return None
    term = str(entry.get("term") or key or "").strip()
    if not term:
        return None
    translations = entry.get("target_translations")
    if not isinstance(translations, dict):
        translations = {}
    translations = {str(k): str(v).strip() for k, v in translations.items() if str(v).strip()}
    row: Dict[str, Any] = {
        "term": term,
        "term_key": str(entry.get("term_key") or key or term.lower()).strip(),
        "target_translations": translations,
        "description": str(entry.get("short_description") or entry.get("description") or "").strip(),
        "source": str(entry.get("source") or "wikidata"),
    }
    if entry.get("rank") is not None:
        try:
            row["rank"] = int(entry["rank"])
        except (TypeError, ValueError):
            pass
    types = entry.get("p31_labels") or entry.get("entity_types")
    if isinstance(types, list):
        row["entity_types"] = [str(t) for t in types if str(t).strip()]
    return row


def _iter_glossary(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield ``(key, entry)`` from a dict-shaped or list-shaped glossary JSON.

    Uses ``ijson`` to stream the multi-GB dict if installed; otherwise falls
    back to ``json.load`` (fine for the smaller sample lists).
    """

    try:
        import ijson  # type: ignore
        with path.open("rb") as handle:
            head = handle.read(1)
        with path.open("rb") as handle:
            if head == b"{":
                for key, entry in ijson.kvitems(handle, ""):
                    yield str(key), entry
            else:
                for entry in ijson.items(handle, "item"):
                    yield "", entry
        return
    except ImportError:
        pass

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key, entry in data.items():
            yield str(key), entry
    elif isinstance(data, list):
        for entry in data:
            yield "", entry


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glossary", required=True, help="Wikidata-derived glossary JSON (dict or list)")
    ap.add_argument("--out", required=True, help="output normalized rows JSONL")
    ap.add_argument("--target-lang", default="zh", help="require a translation in this language")
    ap.add_argument("--limit", type=int, default=0, help="stop after N kept rows (0 = all)")
    args = ap.parse_args()

    src = Path(os.path.expandvars(args.glossary))
    out = Path(os.path.expandvars(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    code = args.target_lang.strip().lower()

    seen, kept = 0, 0
    with out.open("w", encoding="utf-8") as handle:
        for key, entry in _iter_glossary(src):
            seen += 1
            row = normalize_entry(key, entry)
            if row is None or code not in row["target_translations"]:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
            if kept % 100000 == 0:
                print(f"[extract] kept {kept} / seen {seen}", flush=True)
    print(f"[extract] done: kept {kept} rows (with {code}) of {seen} seen -> {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Filter + rank + scale normalized Wikidata term rows into a clean glossary.

Open-world extraction is noisy: without filtering, 1M raw terms would pollute the
term_map with junk (bare numbers, punctuation, Wikimedia meta pages, untranslated
labels). This applies the quality rules, de-duplicates, ranks by popularity, and
truncates to a target scale, emitting a list-shaped glossary JSON that both
``build_maxsim_index.py`` (RASST) and ``build_term_memory_snapshot.py`` accept.

    python scripts/term_memory/filter_terms.py \
        --in  rows.en-zh.jsonl --target-lang zh \
        --limit 100000 --out wiki_open_zh_100k.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_HAS_ALNUM = re.compile(r"[A-Za-z0-9À-￿]")
_META_MARKERS = (
    "disambiguation",
    "list of ",
    "wikimedia category",
    "wikimedia list article",
    "wikimedia disambiguation",
    "template:",
    "category:",
    "wikimedia template",
)
_NUMERIC_ONLY = re.compile(r"^[\d\W_]+$")


def keep_term(row: Dict[str, Any], target_lang: str, *, min_len: int = 2) -> Tuple[bool, str]:
    """Return (keep, reason). ``reason`` names the drop rule when keep is False."""

    term = str(row.get("term") or "").strip()
    if len(term) < min_len:
        return False, "too_short"
    if _NUMERIC_ONLY.match(term):
        return False, "numeric_or_punct"
    if not _HAS_ALNUM.search(term):
        return False, "no_alnum"

    target = str((row.get("target_translations") or {}).get(target_lang) or "").strip()
    if not target:
        return False, "no_translation"

    blob = (term + " " + str(row.get("description") or "")).lower()
    if any(marker in blob for marker in _META_MARKERS):
        return False, "meta_page"

    # For CJK targets an ASCII-identical "translation" usually means untranslated,
    # unless the English term is itself a named entity / acronym worth copying.
    if target_lang in {"zh", "ja"} and not _NON_ASCII.search(target):
        if target.lower() == term.lower() and not (term.isupper() and len(term) <= 6):
            return False, "untranslated_cjk"
    return True, "keep"


def _rank_key(row: Dict[str, Any]) -> Tuple[int, int]:
    """Sort key: lower Wikidata rank == more popular; longer multiword as tiebreak."""
    rank = row.get("rank")
    rank = int(rank) if isinstance(rank, int) else 10**12
    return (rank, -len(str(row.get("term") or "")))


def filter_rows(
    rows: List[Dict[str, Any]], target_lang: str, limit: int = 0
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Filter, de-dup by term_key, rank by popularity, truncate to ``limit``."""

    stats: Dict[str, int] = {}
    seen = set()
    kept: List[Dict[str, Any]] = []
    for row in rows:
        ok, reason = keep_term(row, target_lang)
        if not ok:
            stats[reason] = stats.get(reason, 0) + 1
            continue
        key = str(row.get("term_key") or row.get("term", "")).strip().lower()
        if key in seen:
            stats["dup"] = stats.get("dup", 0) + 1
            continue
        seen.add(key)
        kept.append(row)
    kept.sort(key=_rank_key)
    stats["kept_before_limit"] = len(kept)
    if limit and len(kept) > limit:
        kept = kept[:limit]
    stats["kept"] = len(kept)
    return kept, stats


def _to_glossary_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "term": row["term"],
        "term_key": row.get("term_key") or row["term"].lower(),
        "target_translations": row.get("target_translations") or {},
        "source": row.get("source") or "wikidata",
    }
    if row.get("description"):
        out["short_description"] = row["description"]
    if row.get("rank") is not None:
        out["rank"] = row["rank"]
    return out


def _iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="normalized rows JSONL (from extract_wikidata_terms.py)")
    ap.add_argument("--out", required=True, help="output glossary JSON (list shape)")
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--limit", type=int, default=0, help="keep top-N by popularity (0 = all)")
    args = ap.parse_args()

    rows = list(_iter_rows(Path(os.path.expandvars(args.inp))))
    kept, stats = filter_rows(rows, args.target_lang.strip().lower(), args.limit)
    out = Path(os.path.expandvars(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([_to_glossary_entry(r) for r in kept], ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[filter] {stats}")
    print(f"[filter] wrote {len(kept)} terms -> {out}")


if __name__ == "__main__":
    main()

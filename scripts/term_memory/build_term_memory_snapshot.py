#!/usr/bin/env python3
"""Build an open-memory snapshot (terms.jsonl + manifest) from a glossary JSON.

This is the bridge that turns any source of ``term -> {lang: translation}`` rows
into the snapshot layout the serving agent consumes. The Wikidata pipeline
(``extract_wikidata_terms.py`` -> ``filter_terms.py``) emits the same per-term
shape, so the same builder serves both the curated-glossary and open-world cases.

It writes one ``terms.en-<code>.jsonl`` per target language (one
``framework.agents.term_memory.TermEntry`` per line) and publishes a manifest.
Precomputed ``maxsim`` indexes are NOT built here (that needs the RASST text
encoder on GPU — see ``build_maxsim_index.py``); pass existing index paths via
``--maxsim-index <code>=<path>`` to reuse them, which lets you stand up an
open-memory preset immediately from already-built indexes.

Example (reuse the existing ACL maxsim indexes as an "open" snapshot)::

    python scripts/term_memory/build_term_memory_snapshot.py \
        --glossary /mnt/.../RASST/data/glossaries/acl6060_tagged_gt_raw_min_norm2.json \
        --snapshot-id acl_open_demo --source acl_tagged_gt \
        --langs zh,ja,de \
        --maxsim-index zh=/mnt/.../acl_tagged_raw__zh__lm2/maxsim_...pt \
        --maxsim-index ja=/mnt/.../acl_tagged_raw__ja__lm2/maxsim_...pt \
        --maxsim-index de=/mnt/.../acl_tagged_raw__de__lm2/maxsim_...pt \
        --root $RASST_DEMO_DATA_ROOT/runtime/term_memory
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
from scripts.term_memory.publish_manifest import publish  # noqa: E402


def _iter_glossary_rows(raw: Any) -> Iterable[Dict[str, Any]]:
    """Yield ``{term, target_translations|translation, source, ...}`` rows.

    Accepts the curated list shape (``[{term, target_translations:{zh,...}}]``)
    and a generic ``[{source_label, target_label, target_lang, ...}]`` shape.
    """

    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return
    for row in raw:
        if isinstance(row, dict):
            yield row


def _entries_for_lang(rows: List[Dict[str, Any]], code: str, source: str) -> List[TermEntry]:
    entries: List[TermEntry] = []
    for row in rows:
        term = str(row.get("term") or row.get("source_label") or "").strip()
        if not term:
            continue
        translations = row.get("target_translations")
        if isinstance(translations, dict):
            target = str(translations.get(code) or "").strip()
        else:
            # generic shape: a single translation for one target language
            if str(row.get("target_lang") or "").lower() not in {code, ""}:
                continue
            target = str(row.get("target_label") or row.get("translation") or "").strip()
        if not target:
            continue
        entry = TermEntry(
            term_id=str(row.get("term_id") or row.get("qid") or ""),
            source_lang="en",
            target_lang=code,
            source_label=term,
            target_label=target,
            entity_types=list(row.get("entity_types") or []),
            domains=list(row.get("domains") or []),
            source=str(row.get("source") or source or "glossary"),
            source_url=str(row.get("source_url") or ""),
            updated_at=str(row.get("updated_at") or ""),
        )
        if entry.is_valid():
            entries.append(entry)
    return entries


def _write_jsonl(path: Path, entries: List[TermEntry]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.to_jsonl_line() + "\n")
    return len(entries)


def _parse_kv(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"expected <code>=<path>, got: {item}")
        code, path = item.split("=", 1)
        out[code.strip().lower()] = os.path.expandvars(path.strip())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glossary", required=True, help="source glossary JSON")
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--source", default="glossary", help="provenance tag stored on each term")
    ap.add_argument("--langs", default="zh,ja,de", help="comma-separated target language codes")
    ap.add_argument("--root", required=True, help="term-memory runtime root (holds snapshots/, indexes/, manifests/)")
    ap.add_argument("--preset-id", default="open_wiki_auto",
                    help="expose the snapshot under this preset id (open_wiki_auto, open_wiki_10k, ...)")
    ap.add_argument("--maxsim-index", action="append", default=[], help="<code>=<existing maxsim .pt> to reuse")
    ap.add_argument("--no-require-index", action="store_true", help="publish even if a maxsim index is missing")
    args = ap.parse_args()

    root = Path(os.path.expandvars(args.root))
    snap_dir = root / "snapshots" / args.snapshot_id
    langs = [c.strip().lower() for c in args.langs.split(",") if c.strip()]
    indexes = _parse_kv(args.maxsim_index)

    raw = json.loads(Path(os.path.expandvars(args.glossary)).read_text(encoding="utf-8"))
    rows = list(_iter_glossary_rows(raw))
    print(f"[build] {len(rows)} source rows from {args.glossary}")

    languages: Dict[str, Dict[str, Any]] = {}
    for code in langs:
        entries = _entries_for_lang(rows, code, args.source)
        terms_path = snap_dir / f"terms.en-{code}.jsonl"
        n = _write_jsonl(terms_path, entries)
        key = lang_key(code)
        lang_entry: Dict[str, Any] = {"terms_path": str(terms_path), "num_terms": n, "indexes": {}}
        if code in indexes:
            lang_entry["indexes"]["maxsim"] = indexes[code]
        languages[key] = lang_entry
        print(f"[build] {key}: {n} terms -> {terms_path}" + (f"  (maxsim={indexes[code]})" if code in indexes else "  (no index)"))

    manifest = {
        "snapshot_id": args.snapshot_id,
        "source": args.source,
        "created_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z",
        "root": str(root),
        "languages": languages,
    }
    # Always expose open_wiki_auto (the languages map); also expose under an
    # explicit scale id (open_wiki_10k/100k/1m) when requested, so the UI can
    # select a named scale.
    if args.preset_id and args.preset_id != "open_wiki_auto":
        manifest["scales"] = {args.preset_id: languages}
    archived, current = publish(manifest, root / "manifests", require_index=not args.no_require_index)
    print(f"[build] published snapshot {args.snapshot_id}")
    print(f"[build]   archived: {archived}")
    print(f"[build]   current : {current}")


if __name__ == "__main__":
    main()

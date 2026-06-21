#!/usr/bin/env python3
"""Validate + atomically publish a terminology-memory manifest.

A manifest points the serving agent at one open-memory snapshot (see
``framework/agents/term_memory/manifest.py``). Publishing is the safe swap that
the nightly refresh (Phase E) and the snapshot builder (Phase C) both rely on:

1. validate the manifest parses and its per-language ``maxsim`` indexes exist;
2. write ``<snapshot_id>.json`` into ``manifests/`` (kept for rollback);
3. atomically replace ``manifests/current.json`` (write temp + ``os.replace``)
   so a serving process never observes a half-written file.

Old snapshots are NOT deleted here — a running server may still hold an index
from the previous snapshot until its in-flight requests drain.

Usage::

    # publish a manifest dict supplied as a JSON file, swapping current.json
    python scripts/term_memory/publish_manifest.py \
        --manifest /path/to/wikidata_20260617.json \
        --manifests-dir $RASST_DEMO_DATA_ROOT/runtime/term_memory/manifests

    # repoint current.json at an already-published snapshot (atomic swap only)
    python scripts/term_memory/publish_manifest.py \
        --manifests-dir .../manifests --activate wikidata_20260617
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Make the repo importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework.agents.term_memory.manifest import TermMemoryManifest  # noqa: E402


def validate_manifest(raw: Dict[str, Any], *, base_dir: Path, require_index: bool = True) -> List[str]:
    """Return a list of problems (empty == valid). Raises on unparseable shape."""

    manifest = TermMemoryManifest.from_dict(raw, base_dir=base_dir)
    problems: List[str] = []
    seen = set()
    for preset_id in manifest.preset_ids():
        scope = manifest.languages if preset_id == "open_wiki_auto" else manifest.scales.get(preset_id, {})
        for key, snap in scope.items():
            tag = f"{preset_id}/{key}"
            idx = snap.index_path("maxsim")
            if not idx:
                problems.append(f"{tag}: no maxsim index path")
            elif require_index and not Path(idx).is_file():
                problems.append(f"{tag}: maxsim index missing: {idx}")
            if snap.terms_path and not Path(snap.terms_path).is_file():
                problems.append(f"{tag}: terms file missing: {snap.terms_path}")
            seen.add(tag)
    if not seen:
        problems.append("manifest exposes no (preset, language) snapshots")
    return problems


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def publish(raw: Dict[str, Any], manifests_dir: Path, *, require_index: bool = True) -> Tuple[Path, Path]:
    """Validate, archive ``<snapshot_id>.json``, atomically swap ``current.json``."""

    base_dir = Path(os.path.expandvars(str(raw.get("root")))) if raw.get("root") else manifests_dir.parent
    problems = validate_manifest(raw, base_dir=base_dir, require_index=require_index)
    if problems:
        raise SystemExit("manifest validation failed:\n  - " + "\n  - ".join(problems))
    snapshot_id = str(raw["snapshot_id"])
    archived = manifests_dir / f"{snapshot_id}.json"
    current = manifests_dir / "current.json"
    _atomic_write_json(archived, raw)
    _atomic_write_json(current, raw)
    return archived, current


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests-dir", required=True, help="directory holding current.json + archived snapshots")
    ap.add_argument("--manifest", help="path to a manifest JSON to validate + publish")
    ap.add_argument("--activate", help="snapshot_id of an already-archived manifest to make current")
    ap.add_argument("--no-require-index", action="store_true", help="don't fail if maxsim index files are missing")
    args = ap.parse_args()

    manifests_dir = Path(os.path.expandvars(args.manifests_dir))
    require_index = not args.no_require_index

    if args.activate:
        archived = manifests_dir / f"{args.activate}.json"
        if not archived.is_file():
            raise SystemExit(f"no archived manifest: {archived}")
        raw = json.loads(archived.read_text(encoding="utf-8"))
        _, current = publish(raw, manifests_dir, require_index=require_index)
        print(f"activated snapshot {args.activate} -> {current}")
        return

    if not args.manifest:
        raise SystemExit("provide --manifest <file> or --activate <snapshot_id>")
    raw = json.loads(Path(os.path.expandvars(args.manifest)).read_text(encoding="utf-8"))
    archived, current = publish(raw, manifests_dir, require_index=require_index)
    print(f"published snapshot {raw.get('snapshot_id')}")
    print(f"  archived: {archived}")
    print(f"  current : {current}")


if __name__ == "__main__":
    main()

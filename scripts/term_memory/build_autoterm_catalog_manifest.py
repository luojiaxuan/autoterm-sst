#!/usr/bin/env python3
"""Create and validate a candidate manifest for an AutoTerm catalog."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.domain_taxonomy import (
    DOMAIN_TO_PRESET,
    WORKING_DOMAINS,
    WORKING_PRESET_META,
)
from scripts.term_memory.publish_manifest import validate_manifest


def _count_rows(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return len(payload)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-id", required=True)
    ap.add_argument("--catalog-dir", required=True, type=Path)
    ap.add_argument("--index-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--source", default="Wikidata/RDF-derived RealSI-domain catalog")
    ap.add_argument("--include-merged", action="store_true")
    args = ap.parse_args()

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    unknown = [item for item in domains if item not in DOMAIN_TO_PRESET]
    if unknown:
        raise SystemExit(f"unknown domains: {', '.join(unknown)}")
    lang_key = f"en-{args.target_lang}"
    scales: Dict[str, Dict[str, Any]] = {}
    preset_meta: Dict[str, Dict[str, Any]] = {}
    for domain in domains:
        preset = DOMAIN_TO_PRESET[domain]
        glossary = args.catalog_dir / f"{preset}.json"
        index = args.index_root / preset / lang_key / "maxsim.pt"
        count = _count_rows(glossary)
        scales[preset] = {
            lang_key: {
                "terms_path": str(glossary),
                "indexes": {"maxsim": str(index)},
                "num_terms": count,
            }
        }
        meta = dict(WORKING_PRESET_META[preset])
        meta.update(
            {
                "id": preset,
                "preset_id": preset,
                "domain_id": domain,
                "fallback_preset_id": "",
                "term_count": count,
                "maxsim_index_path": str(index),
                "enabled_for_auto_router": True,
            }
        )
        preset_meta[preset] = meta

    if args.include_merged:
        candidates = sorted(args.catalog_dir.glob(f"merged_realsi_*_{args.target_lang}.json"))
        if len(candidates) != 1:
            raise SystemExit(f"expected one merged glossary, found {len(candidates)}")
        preset = candidates[0].stem.removesuffix(f"_{args.target_lang}")
        index = args.index_root / preset / lang_key / "maxsim.pt"
        count = _count_rows(candidates[0])
        scales[preset] = {
            lang_key: {
                "terms_path": str(candidates[0]),
                "indexes": {"maxsim": str(index)},
                "num_terms": count,
            }
        }
        preset_meta[preset] = {
            "label": f"Merged RealSI-domain glossary {count // 1000}k",
            "domain": "merged",
            "domain_id": "merged",
            "term_count": count,
            "maxsim_index_path": str(index),
            "enabled_for_auto_router": False,
        }

    root = args.catalog_dir.parent
    manifest = {
        "snapshot_id": args.snapshot_id,
        "source": args.source,
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "root": str(root),
        "scales": scales,
        "preset_meta": preset_meta,
    }
    problems = validate_manifest(manifest, base_dir=root, require_index=True)
    if problems:
        raise SystemExit("manifest validation failed:\n  - " + "\n  - ".join(problems))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"validated {len(scales)} presets -> {args.out}")


if __name__ == "__main__":
    main()

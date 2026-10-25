#!/usr/bin/env python3
"""Build unrouted union baselines from aligned slice prefixes and write a serving manifest.

Each union concatenates the glossary rows, MaxSim ``term_list`` rows and ``text_embs`` rows of
its member slices in the given order. Only exact duplicate pairs (same normalized English
source and same target translation) are dropped; conflicting translations of one source are
kept, matching the merged-glossary policy of the ten-talk evaluation. No term is re-encoded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.source_normalization import normalize_english_source  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zh_target(entry: dict) -> str:
    value = (entry.get("target_translations") or {}).get("zh")
    if isinstance(value, list):
        value = value[0] if value else ""
    return " ".join(str(value or "").split())


def build_union(prefix_root: Path, members: list[str], preset_id: str, out_root: Path) -> dict:
    rows, term_list, embs, seen, dropped = [], [], [], set(), 0
    for member in members:
        glossary = json.loads((prefix_root / member / "glossary.json").read_text(encoding="utf-8"))
        index = torch.load(prefix_root / member / "en-zh" / "maxsim.pt", map_location="cpu", weights_only=False)
        assert len(glossary) == len(index["term_list"]) == index["text_embs"].shape[0], member
        for entry, row, emb in zip(glossary, index["term_list"], index["text_embs"]):
            assert normalize_english_source(entry["term"]) == normalize_english_source(row["term"]), member
            key = (normalize_english_source(entry["term"]), zh_target(entry))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            rows.append(entry)
            term_list.append(row)
            embs.append(emb)
    target = out_root / preset_id
    (target / "en-zh").mkdir(parents=True, exist_ok=True)
    glossary_path = target / "glossary.json"
    index_path = target / "en-zh" / "maxsim.pt"
    glossary_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    torch.save(
        {
            "text_embs": torch.stack(embs),
            "term_list": term_list,
            "build_metadata": {
                "builder": "scripts/term_memory/build_slice_union_baselines.py",
                "preset_id": preset_id,
                "members": members,
                "dedup_rule": "drop exact (normalized source, zh target) duplicates; keep conflicting targets",
                "dropped_exact_duplicates": dropped,
                "term_count": len(rows),
            },
        },
        index_path,
    )
    return {
        "preset_id": preset_id,
        "members": members,
        "term_count": len(rows),
        "dropped_exact_duplicates": dropped,
        "glossary_path": str(glossary_path),
        "glossary_sha256": sha256(glossary_path),
        "maxsim_index_path": str(index_path),
        "maxsim_index_sha256": sha256(index_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-root", type=Path, required=True)
    ap.add_argument("--union", action="append", required=True, metavar="PRESET_ID=MEMBER,MEMBER")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--manifest-out", type=Path, required=True)
    ap.add_argument("--snapshot-id", required=True)
    args = ap.parse_args()

    fragment = json.loads((args.prefix_root / "manifest_fragment.json").read_text(encoding="utf-8"))
    scales, preset_meta = {}, {}
    for preset_id, meta in fragment["preset_meta"].items():
        scales[preset_id] = {"en-zh": {"terms_path": meta["glossary_path"], "indexes": {"maxsim": meta["maxsim_index_path"]}, "num_terms": meta["term_count"]}}
        preset_meta[preset_id] = meta
    report = []
    for spec in args.union:
        preset_id, members = spec.split("=", 1)
        info = build_union(args.prefix_root, members.split(","), preset_id, args.out_root)
        report.append(info)
        scales[preset_id] = {"en-zh": {"terms_path": info["glossary_path"], "indexes": {"maxsim": info["maxsim_index_path"]}, "num_terms": info["term_count"]}}
        preset_meta[preset_id] = {
            "label": f"Unrouted union of {len(info['members'])} slices",
            "domain": "merged",
            "domain_id": "merged",
            "id": preset_id,
            "preset_id": preset_id,
            "fallback_preset_id": "",
            "term_count": info["term_count"],
            "maxsim_index_path": info["maxsim_index_path"],
            "enabled_for_auto_router": False,
        }
    manifest = {
        "snapshot_id": args.snapshot_id,
        "source": "aligned 1k prefixes of the ten-domain deepcat catalog plus unrouted unions",
        "created_at": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "root": str(args.out_root),
        "scales": scales,
        "preset_meta": preset_meta,
    }
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

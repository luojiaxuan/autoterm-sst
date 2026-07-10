#!/usr/bin/env python3
"""Build a fixed-size glossary as GT terms UNION deterministic fillers.

Motivation (2026-07-08): a glossary-channel eval is meaningless when the gold
terms are not in the retrieval inventory at all. The 3-talk long-stream rerun
scored `medicine_core_10k` (broad wiki_medicine) against the hard medicine
gold and its exact gold coverage was 1/54 — identical to fixed NLP — so
term_acc measured base-model retention, not retrieval. The ACL side already
solved this with `acl6060_tagged_gt_union_gs10000` (238 GT + 9,762 wiki
fillers): the GT entries are byte-identical at every scale, so the gold
denominator is fixed and only the distractor count varies.

This script generalizes that recipe:

    python scripts/term_memory/build_gt_union_gs_glossary.py \
        --gt hard_medicine_gt_raw_unique212.json \
        --filler wiki_glossary_medicine_enriched.json \
        --size 10000 --target-lang zh --seed 1215 \
        --out medicine_hardraw_gt_union_gs10000.json

Guarantees:
  * every GT entry is included unchanged (dict passthrough, GT-first order);
  * GT wins collisions with fillers on the normalized source term;
  * fillers must carry a non-empty target-lang translation;
  * filler selection is deterministic: sort by (term, translation), then
    seeded shuffle, then take exactly `size - len(GT)`;
  * output length == --size, or the script fails loudly (no silent shortfall).

The output is list-shaped glossary JSON accepted by RASST
``build_maxsim_index.py`` and ``build_term_memory_snapshot.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_WS = re.compile(r"\s+")


def norm_term(term: str) -> str:
    return _WS.sub(" ", str(term).strip()).casefold()


def load_glossary(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a list-shaped glossary JSON, got {type(data)}")
    return data


def translation_of(entry: Dict[str, Any], target_lang: str) -> str:
    translations = entry.get("target_translations")
    if isinstance(translations, dict):
        value = translations.get(target_lang)
        if value:
            return str(value)
    # tolerate flat schemas: {"term": ..., "translation": ...}
    value = entry.get("translation") or entry.get(target_lang)
    return str(value) if value else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", required=True, type=Path, help="GT glossary JSON (kept byte-identical)")
    parser.add_argument("--filler", required=True, type=Path, help="filler pool glossary JSON")
    parser.add_argument("--size", required=True, type=int, help="exact output size, e.g. 10000")
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--seed", type=int, default=1215)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None, help="optional JSON report path (default: <out>.report.json)")
    args = parser.parse_args()

    gt = load_glossary(args.gt)
    filler_pool = load_glossary(args.filler)

    gt_keys = set()
    for entry in gt:
        key = norm_term(entry.get("term", ""))
        if not key:
            raise SystemExit(f"GT entry without a term: {entry!r}")
        if key in gt_keys:
            raise SystemExit(f"duplicate GT term after normalization: {key!r}")
        if not translation_of(entry, args.target_lang):
            raise SystemExit(f"GT term missing {args.target_lang} translation: {entry.get('term')!r}")
        gt_keys.add(key)

    if len(gt) >= args.size:
        raise SystemExit(f"--size {args.size} must exceed GT size {len(gt)}")

    seen = set(gt_keys)
    candidates: List[Dict[str, Any]] = []
    skipped_no_translation = 0
    skipped_collision = 0
    for entry in filler_pool:
        key = norm_term(entry.get("term", ""))
        if not key:
            continue
        if key in seen:
            skipped_collision += 1
            continue
        if not translation_of(entry, args.target_lang):
            skipped_no_translation += 1
            continue
        seen.add(key)
        candidates.append(entry)

    need = args.size - len(gt)
    if len(candidates) < need:
        raise SystemExit(
            f"filler pool too small: need {need} unique {args.target_lang}-translated fillers, "
            f"have {len(candidates)} (skipped: {skipped_collision} GT/dup collisions, "
            f"{skipped_no_translation} without {args.target_lang})"
        )

    candidates.sort(key=lambda e: (norm_term(e.get("term", "")), translation_of(e, args.target_lang)))
    random.Random(args.seed).shuffle(candidates)
    fillers = candidates[:need]

    merged = list(gt) + fillers
    assert len(merged) == args.size, (len(merged), args.size)
    assert len({norm_term(e["term"]) for e in merged}) == args.size, "normalized-term collision in output"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    report = {
        "out": str(args.out),
        "size": args.size,
        "gt_file": str(args.gt),
        "gt_entries": len(gt),
        "filler_file": str(args.filler),
        "filler_pool": len(filler_pool),
        "filler_used": len(fillers),
        "filler_skipped_gt_or_dup_collision": skipped_collision,
        "filler_skipped_no_translation": skipped_no_translation,
        "target_lang": args.target_lang,
        "seed": args.seed,
        "gt_coverage_in_output": 1.0,
    }
    report_path = args.report or args.out.with_suffix(args.out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())

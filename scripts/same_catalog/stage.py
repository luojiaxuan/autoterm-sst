"""Write the compact input bundle: ten 10k slices and the general pairs that fill the catalog to 1M."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from catalog import distinct, fill, pair_of

DOMAINS = ("nlp", "medicine", "education", "finance", "legal", "environment", "entertainment", "science", "sports", "art")


def rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for entry in data.values() if isinstance(data, dict) else data:
        zh = (entry.get("target_translations") or {}).get("zh", entry.get("translation"))
        term = str(entry.get("term", "")).strip()
        if term and zh:
            yield {"term": term, "zh": str(zh).strip(), "desc": str(entry.get("short_description") or "")}


def write(path: Path, data) -> str:
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as out:
        json.dump(data, out, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog-dir", type=Path, required=True)
    ap.add_argument("--prefix-dir", type=Path, required=True)
    ap.add_argument("--general", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1_000_000)
    args = ap.parse_args()

    slices = {d: distinct(rows(args.catalog_dir / f"{d}_core_10k.json")) for d in DOMAINS}
    for domain, entries in slices.items():
        core = {pair_of(e) for e in rows(args.prefix_dir / f"{domain}_core_1k" / "glossary.json")}
        if core - {pair_of(e) for e in entries}:
            raise ValueError(f"{domain}: the 1k entries are not contained in the 10k slice")
    union = distinct(e for d in DOMAINS for e in slices[d])
    added = fill(union, rows(args.general), args.size)

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "domains": list(DOMAINS),
        "slice_pairs": {d: len(slices[d]) for d in DOMAINS},
        "union_pairs": len(union),
        "added_pairs": len(added),
        "total_pairs": len(union) + len(added),
        "sha256": {
            "slices.json.gz": write(args.out / "slices.json.gz", slices),
            "added.json.gz": write(args.out / "added.json.gz", added),
        },
    }
    (args.out / "bundle.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

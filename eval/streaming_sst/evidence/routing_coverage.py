"""Live routing coverage of finished streaming runs.

For every decoded chunk, the correct slice is the slice mapped to the expected domain of its talk.
Coverage is the share of chunks whose correct slice is in the working set the chunk retrieved from;
top-1 is the share whose top slice is correct; top changes counts how often the top slice changed.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_JSON")
    ap.add_argument("--slice-map", required=True, metavar="DOMAIN=SLICE,...")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    slice_of = dict(x.split("=", 1) for x in args.slice_map.split(","))
    result = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        records = json.load(open(path, encoding="utf-8"))["records"]
        cover = top = changes = 0
        prev = None
        for r in records:
            gold = slice_of[str(r["expected_domain"]).lower()]
            cover += gold in r["selected_slice_presets"]
            top += r["active_preset"] == gold
            changes += prev is not None and r["active_preset"] != prev
            prev = r["active_preset"]
        result[name] = {"chunks": len(records), "cover": round(100 * cover / len(records), 2),
                        "top1": round(100 * top / len(records), 2), "top_changes": changes}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

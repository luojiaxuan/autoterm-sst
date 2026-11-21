"""Live routing coverage of finished streaming runs.

For every decoded chunk, the correct slice is the slice mapped to the expected domain of its talk.
Coverage is the share of chunks whose correct slice is in the working set the chunk retrieved from.
The top slice is the first slice of that working set: the posterior argmax of the language-model
router, or the pinned confirmed slice of the similarity router; a chunk decoded before the first
routing update has an empty working set and no top slice. Top-1 is the share of chunks whose
top slice is correct, and top changes counts how often the top slice changed.

Per talk, the delay to top is the audio time from the talk's first sample to the end of the first
chunk whose top slice is the talk's slice; a talk where that never happens has no delay. A group
pools the talks of several runs: its median delay is over the talks that reach the top, and
never_top counts the rest.
"""

import argparse
import json
import statistics
from pathlib import Path

SAMPLE_RATE = 16000


def talk_delays(payload, slice_of):
    delays = []
    for span in payload["block_spans"]:
        lo, hi = int(span["start_sample"]), int(span["end_sample"])
        gold = slice_of[str(span["expected_domain"]).lower()]
        hit = next((int(r["cursor_samples"]) for r in payload["records"]
                    if lo < int(r["cursor_samples"]) <= hi and r["selected_slice_presets"][:1] == [gold]), None)
        delays.append(None if hit is None else round((hit - lo) / SAMPLE_RATE, 2))
    return delays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_JSON")
    ap.add_argument("--slice-map", required=True, metavar="DOMAIN=SLICE,...")
    ap.add_argument("--group", action="append", default=[], metavar="LABEL=NAME,NAME,...",
                    help="pool the talks of these runs for the delay to top")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    slice_of = dict(x.split("=", 1) for x in args.slice_map.split(","))
    result = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        payload = json.load(open(path, encoding="utf-8"))
        records = payload["records"]
        cover = top = changes = 0
        prev = None
        for r in records:
            gold = slice_of[str(r["expected_domain"]).lower()]
            working = r["selected_slice_presets"]
            first = working[0] if working else None
            cover += gold in working
            top += first == gold
            changes += prev is not None and first != prev
            prev = first
        result[name] = {"chunks": len(records), "cover": round(100 * cover / len(records), 2),
                        "top1": round(100 * top / len(records), 2), "top_changes": changes,
                        "talk_delay_to_top_s": talk_delays(payload, slice_of)}
    for spec in args.group:
        label, names = spec.split("=", 1)
        names = names.split(",")
        delays = [d for n in names for d in result[n]["talk_delay_to_top_s"]]
        reached = [d for d in delays if d is not None]
        result[label] = {"talks": len(delays), "never_top": len(delays) - len(reached),
                         "median_delay_to_top_s_among_reached": round(statistics.median(reached), 2),
                         "cover": round(sum(result[n]["cover"] for n in names) / len(names), 2),
                         "top1": round(sum(result[n]["top1"] for n in names) / len(names), 2)}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

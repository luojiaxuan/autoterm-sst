"""Vector timeline of routing correctness on a multi-domain stream, one row per router.

Each talk is drawn at equal width under its domain name. A router's row colors every decoded
chunk by the state of the talk's own slice: the top slice of the working set (the posterior
argmax of the language-model router, or the pinned confirmed slice of the similarity router),
lower in the working set, or absent from it.
"""

import argparse
import json
from pathlib import Path

STATES = [("top", "#2f855a", "Correct slice on top"), ("in", "#9fd8b5", "Correct slice lower in working set"),
          ("out", "#e58f8f", "Correct slice absent")]
COLOR = {k: c for k, c, _ in STATES}
X0, WIDTH, ROW_H = 330, 1440, 44


def state(record, gold):
    working = record["selected_slice_presets"]
    if working and working[0] == gold:
        return "top"
    return "in" if gold in working else "out"


def runs(payload, slice_of):
    """Yields [x_start, x_end, color] runs with each talk scaled to equal width."""
    spans = payload["block_spans"]
    talk_w = WIDTH / len(spans)
    out = []
    for k, s in enumerate(spans):
        lo, hi = int(s["start_sample"]), int(s["end_sample"])
        gold = slice_of[str(s["expected_domain"]).lower()]
        for r in payload["records"]:
            c = int(r["cursor_samples"])
            if not lo < c <= hi:
                continue
            a = X0 + k * talk_w + (max(int(r["start_sample"]), lo) - lo) / (hi - lo) * talk_w
            b = X0 + k * talk_w + (c - lo) / (hi - lo) * talk_w
            color = COLOR[state(r, gold)]
            if out and out[-1][2] == color and abs(out[-1][1] - a) < 1e-6:
                out[-1][1] = b
            else:
                out.append([a, b, color])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", action="append", required=True, metavar="LABEL=RUN_JSON",
                    help="row label and run JSON, in display order")
    ap.add_argument("--slice-map", required=True, metavar="DOMAIN=SLICE,...")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    slice_of = dict(x.split("=", 1) for x in args.slice_map.split(","))
    routers = [spec.split("=", 1) for spec in args.router]
    payloads = [json.load(open(p, encoding="utf-8")) for _, p in routers]
    spans = [(s["start_sample"], s["end_sample"]) for s in payloads[0]["block_spans"]]
    assert all([(s["start_sample"], s["end_sample"]) for s in p["block_spans"]] == spans for p in payloads)
    talk_w = WIDTH / len(spans)
    domains = [str(s["expected_domain"]).lower() for s in payloads[0]["block_spans"]]

    y = 70
    parts = [f'<text class="row-label" x="30" y="{y + 28}">Talk domain</text>',
             f'<rect class="talks" x="{X0}" y="{y}" width="{WIDTH}" height="{ROW_H}"/>',
             '<g class="inside">' + "".join(
                 f'<text x="{X0 + (k + .5) * talk_w:.2f}" y="{y + 28}">{"NLP" if d == "nlp" else d.capitalize()}</text>'
                 for k, d in enumerate(domains)) + '</g>']
    y += ROW_H + 18
    for (label, _), payload in zip(routers, payloads):
        rects = "".join(f'<rect x="{a:.2f}" y="{y}" width="{b - a:.2f}" height="{ROW_H}" fill="{c}"/>'
                        for a, b, c in runs(payload, slice_of))
        parts.append(f'<text class="row-label" x="30" y="{y + 28}">{label}</text>')
        parts.append(f'<g class="block">{rects}</g><rect class="outline" x="{X0}" y="{y}" width="{WIDTH}" height="{ROW_H}"/>')
        y += ROW_H + 18
    grid = "".join(f'<line x1="{X0 + k * talk_w:.2f}" y1="62" x2="{X0 + k * talk_w:.2f}" y2="{y - 10}"/>'
                   for k in range(1, len(spans)))
    legend = "".join(f'<rect x="{X0 + 420 * i}" y="18" width="23" height="23" rx="4" fill="{c}"/>'
                     f'<text class="legend" x="{X0 + 32 + 420 * i}" y="36">{t}</text>' for i, (_, c, t) in enumerate(STATES))
    height = y + 4
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="{height}" viewBox="0 0 1800 {height}">
<rect width="1800" height="{height}" fill="#ffffff"/>
<style>
text {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; fill: #344054; }}
.legend {{ font-size: 20px; font-weight: 500; }}
.row-label {{ font-size: 22px; font-weight: 600; }}
.inside text {{ font-size: 17px; font-weight: 600; fill: #344054; text-anchor: middle; }}
.grid line {{ stroke: #c3ccd8; stroke-width: 2; shape-rendering: crispEdges; }}
.talks {{ fill: #eef2f6; stroke: #d4dce7; stroke-width: 2; shape-rendering: crispEdges; }}
.outline {{ fill: none; stroke: #d4dce7; stroke-width: 2; shape-rendering: crispEdges; }}
.block {{ shape-rendering: crispEdges; }}
</style>
{legend}
<g class="grid">{grid}</g>
{"".join(parts)}
</svg>
'''
    args.out.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

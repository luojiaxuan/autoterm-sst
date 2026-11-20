"""Vector routing timeline of the ten-talk stream for one or more routers.

Each talk is drawn at equal width. The first row is the talk domain; each router
adds a row for its top slice (the posterior argmax of the language-model router or
the confirmed slice of the similarity router, both stored as active_preset) and a
row marking whether the talk's own slice is in the working set.
"""

import argparse
import json
from pathlib import Path

COLORS = {"nlp": "#4f7cac", "medicine": "#d9822b", "other": "#94a3b8", "in": "#2f855a", "out": "#e8a1a1"}
GOLD = {"nlp": "nlp_core_1k", "medicine": "medicine_core_1k"}
X0, WIDTH, ROW_H = 400, 1380, 46


def domain(preset):
    p = str(preset).lower()
    return "nlp" if p.startswith("nlp_") else "medicine" if p.startswith("medicine_") else "other"


def segments(payload, value):
    """Yields (x_start, x_end, color) runs, each talk scaled to equal width."""
    spans = payload["block_spans"]
    talk_w = WIDTH / len(spans)
    runs = []
    for k, s in enumerate(spans):
        lo, hi = int(s["start_sample"]), int(s["end_sample"])
        gold = str(s["expected_domain"]).lower()
        for r in payload["records"]:
            c = int(r["cursor_samples"])
            if not lo < c <= hi:
                continue
            a = X0 + k * talk_w + (max(int(r["start_sample"]), lo) - lo) / (hi - lo) * talk_w
            b = X0 + k * talk_w + (c - lo) / (hi - lo) * talk_w
            color = COLORS[value(r, gold)]
            if runs and runs[-1][2] == color and abs(runs[-1][1] - a) < 1e-6:
                runs[-1][1] = b
            else:
                runs.append([a, b, color])
    return runs


def row(y, runs):
    rects = "".join(f'<rect x="{a:.2f}" y="{y}" width="{b - a:.2f}" height="{ROW_H}" fill="{c}"/>' for a, b, c in runs)
    return f'<g class="block">{rects}</g><rect class="outline" x="{X0}" y="{y}" width="{WIDTH}" height="{ROW_H}"/>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", action="append", required=True, metavar="LABEL=RUN_JSON",
                    help="row-group label and run JSON, in display order")
    ap.add_argument("--top-label", action="append", required=True, help="label of each router's top-slice row")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    routers = [spec.split("=", 1) for spec in args.router]
    payloads = [json.load(open(p, encoding="utf-8")) for _, p in routers]
    spans = [(s["start_sample"], s["end_sample"]) for s in payloads[0]["block_spans"]]
    assert all([(s["start_sample"], s["end_sample"]) for s in p["block_spans"]] == spans for p in payloads)

    talk_w = WIDTH / len(spans)
    domains = [str(s["expected_domain"]).lower() for s in payloads[0]["block_spans"]]
    y = 70
    parts = []
    talk_rects = "".join(f'<rect x="{X0 + k * talk_w:.2f}" y="{y}" width="{talk_w:.2f}" height="{ROW_H}" fill="{COLORS[d]}"/>'
                         for k, d in enumerate(domains))
    names = "".join(f'<text x="{X0 + (k + .5) * talk_w:.2f}" y="{y + 30}">{"NLP" if d == "nlp" else "Medicine"}</text>'
                    for k, d in enumerate(domains))
    parts.append(f'<text class="row-label" x="40" y="{y + 30}">Talk domain</text>')
    parts.append(f'<g class="block">{talk_rects}</g><rect class="outline" x="{X0}" y="{y}" width="{WIDTH}" height="{ROW_H}"/>'
                 f'<g class="inside">{names}</g>')
    y += ROW_H + 34
    for (label, _), top_label, payload in zip(routers, args.top_label, payloads):
        parts.append(f'<text class="group-label" x="40" y="{y + 10}">{label}</text>')
        y += 22
        parts.append(f'<text class="row-label" x="60" y="{y + 30}">{top_label}</text>')
        parts.append(row(y, segments(payload, lambda r, g: domain((r["selected_slice_presets"] or [r["active_preset"]])[0]))))
        y += ROW_H + 10
        parts.append(f'<text class="row-label" x="60" y="{y + 30}">Correct slice in working set</text>')
        parts.append(row(y, segments(payload, lambda r, g: "in" if GOLD[g] in r["selected_slice_presets"] else "out")))
        y += ROW_H + 30
    grid = "".join(f'<line x1="{X0 + k * talk_w:.2f}" y1="60" x2="{X0 + k * talk_w:.2f}" y2="{y - 20}"/>' for k in range(1, len(spans)))
    ticks = "".join(f'<text x="{X0 + (k + .5) * talk_w:.2f}" y="{y}">Talk {k + 1}</text>' for k in range(len(spans)))
    legend_items = [("nlp", "NLP"), ("medicine", "Medicine"), ("other", "Other"), ("in", "Correct slice in set"), ("out", "Absent")]
    legend = "".join(f'<rect x="{560 + 230 * i}" y="20" width="23" height="23" rx="4" fill="{COLORS[k]}"/>'
                     f'<text class="legend" x="{592 + 230 * i}" y="38">{t}</text>' for i, (k, t) in enumerate(legend_items))
    height = y + 20
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="{height}" viewBox="0 0 1800 {height}">
<rect width="1800" height="{height}" fill="#ffffff"/>
<style>
text {{ font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; fill: #344054; }}
.legend {{ font-size: 20px; font-weight: 500; }}
.group-label {{ font-size: 23px; font-weight: 700; }}
.row-label {{ font-size: 21px; font-weight: 500; }}
.inside text {{ font-size: 19px; font-weight: 700; fill: #ffffff; text-anchor: middle; }}
.tick text {{ font-size: 18px; font-weight: 500; fill: #607089; text-anchor: middle; }}
.grid line {{ stroke: #d4dce7; stroke-width: 2; shape-rendering: crispEdges; }}
.outline {{ fill: none; stroke: #d4dce7; stroke-width: 2; shape-rendering: crispEdges; }}
.block {{ shape-rendering: crispEdges; }}
</style>
{legend}
<g class="grid">{grid}</g>
{"".join(parts)}
<g class="tick">{ticks}</g>
</svg>
'''
    args.out.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()

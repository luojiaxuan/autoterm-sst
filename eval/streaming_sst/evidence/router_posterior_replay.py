"""Replay the language-model router's posterior on the translations of a recorded run.

The router text of a chunk is the last four non-empty translations, newline-joined, as on the
server. Scores refresh when that text has changed and at least 5 seconds of audio have passed
since the last refresh; each refresh takes one sticky-HMM filter step weighted by the share of
new chunks in the four-chunk context. Slice prototypes are the run's manifest descriptions plus
the server's built-in router prototypes, scored with the server's LanguageModelSliceScorer. The
output lists, for every record, the four most probable slices with their posterior probability,
and how often that order matches the working set the live run recorded.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from framework.agents.omni import DOMAIN_ROUTER_PROTOTYPES  # noqa: E402
from framework.agents.term_memory.language_model_router import (  # noqa: E402
    LanguageModelSliceScorer,
    filter_step,
)

WINDOW = 4
UPDATE_S = 5.0
SWITCH_RATE = 1e-3


def prototypes_from(meta):
    out = {}
    for preset, item in sorted(meta.items()):
        domain = item["domain"]
        texts = [item.get("description") or ""] + list(DOMAIN_ROUTER_PROTOTYPES.get(domain, ()))
        texts = [t.strip() for t in texts if t and t.strip()]
        out.setdefault(domain, [])
        out[domain] += [t for t in texts if t not in out[domain]]
    return out


def router_texts(records):
    history, texts = [], []
    for record in records:
        current = str(record.get("text") or "").strip()
        if current and (not history or history[-1] != current):
            history.append(current)
        history = history[-WINDOW:]
        texts.append("\n".join(history).strip())
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="run JSON with a records list")
    ap.add_argument("--slices", type=Path, required=True, help="manifest preset_meta: preset -> domain, description")
    ap.add_argument("--domains", required=True, help="comma-separated slice domains of the catalog")
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--end-event", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    records = [r for r in json.loads(args.run.read_text(encoding="utf-8"))["records"]
               if r["event_idx"] <= args.end_event]
    domains = args.domains.split(",")
    meta = {k: v for k, v in json.loads(args.slices.read_text(encoding="utf-8")).items() if v["domain"] in domains}
    scorer = LanguageModelSliceScorer(model_id=args.model, device=args.device, batch_size=8,
                                      prototypes=prototypes_from(meta), target_language="Chinese")
    scorer._load()

    texts = router_texts(records)
    refresh, scored_text, scored_at = [], None, None
    for t, (text, record) in enumerate(zip(texts, records)):
        now = float(record["cursor_s"])
        due = bool(text) and (scored_text is None or (text != scored_text and now - scored_at >= UPDATE_S))
        if due:
            scored_text, scored_at = text, now
            refresh.append(t)
    scores = scorer._score_batch_sync([texts[t] for t in refresh], domains)
    by_chunk = dict(zip(refresh, scores))

    posterior, last, rows, agree = {}, None, [], 0
    for t, record in enumerate(records):
        if t in by_chunk:
            weight = 1.0 if last is None else min(1.0, (t - last) / WINDOW)
            posterior = filter_step(posterior, by_chunk[t], switch_rate=SWITCH_RATE, evidence_weight=weight)
            last = t
        ranked = sorted(posterior, key=lambda d: -posterior[d])[:4]
        live = [p.split("_core")[0] for p in record.get("selected_slice_presets") or []]
        agree += ranked == live[:4]
        rows.append({"event_idx": record["event_idx"],
                     "top_slices": [{"domain": d, "preset": f"{d}_core_1k", "probability": round(posterior[d], 4)}
                                    for d in ranked],
                     "live_working_set": live})
    result = {"window": WINDOW, "update_s": UPDATE_S, "switch_rate": SWITCH_RATE, "model": args.model,
              "refreshes": len(refresh), "records": len(records),
              "working_set_order_matches_live": round(agree / len(records), 4), "rows": rows}
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()

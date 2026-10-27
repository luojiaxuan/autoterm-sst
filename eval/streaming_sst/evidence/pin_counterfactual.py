"""Counterfactual for the pinning rule on the audited ten-talk AutoTerm run.

Raw context-similarity scores are recomputed open-loop exactly as in router_replay.py
(window w=4, original prototypes). For every window the live run's confirmed slice
(``active_preset``) is taken from the record, and two working sets are compared on the
same raw scores: without pinning, the four highest-scoring slices; with pinning, the
confirmed slice plus the three highest-scoring others (all slices hold 1k terms, so this
is what select_budgeted_slices builds under the 4-slice / 4k-term budget).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from router_replay import VARIANTS, centroids, encode

SAMPLE_RATE = 16000


def runs(mask):
    lengths, current = [], 0
    for value in mask:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    run = json.load(open(args.run, encoding="utf-8"))
    recs = run["records"]
    n = len(recs)
    gold = [str(r["expected_domain"]).lower() for r in recs]
    active = [str(r["active_preset"]).removesuffix("_core_1k") for r in recs]
    live_sets = [{p.removesuffix("_core_1k") for p in (r["selected_slice_presets"] or [])} for r in recs]
    texts = [str(r.get("text") or "") for r in recs]

    ctx = ["".join(texts[max(0, t - args.window):t]) for t in range(n)]
    emb = encode(model, tok, [c if c.strip() else " " for c in ctx], device)
    for t in range(1, n):
        if not ctx[t].strip():
            emb[t] = emb[t - 1]
    names, c = centroids(model, tok, VARIANTS["original"], device)
    order = np.argsort(-(emb @ c.T), axis=1)
    ranked = [[names[i] for i in row] for row in order]

    no_pin = [set(row[:4]) for row in ranked]
    pinned = [{a} | set([d for d in row if d != a][:3]) for a, row in zip(active, ranked)]
    valid = np.arange(n) >= 1
    in_no_pin = np.array([g in s for g, s in zip(gold, no_pin)])
    in_pinned = np.array([g in s for g, s in zip(gold, pinned)])
    in_live = np.array([g in s for g, s in zip(gold, live_sets)])
    active_correct = np.array([a == g for a, g in zip(active, gold)])
    active_dropped = np.array([a not in s for a, s in zip(active, no_pin)])
    rescued = active_correct & active_dropped & valid

    boundaries = [span["start_sample"] / SAMPLE_RATE for span in run["block_spans"][1:]]
    cursor = np.array([float(r["cursor_s"]) for r in recs])
    since_boundary = np.array([
        min([cursor[t] - b for b in boundaries if b <= cursor[t]], default=np.inf) for t in range(n)
    ])
    episodes = runs(rescued)

    result = {
        "run": str(args.run),
        "window_chunks": args.window,
        "windows": int(valid.sum()),
        "device": device,
        "gold_in_working_set": {
            "no_pin_raw_top4": round(float(in_no_pin[valid].mean()), 4),
            "pin_replay": round(float(in_pinned[valid].mean()), 4),
            "live_run": round(float(in_live[valid].mean()), 4),
        },
        "confirmed_slice_correct": round(float(active_correct[valid].mean()), 4),
        "confirmed_slice_outside_raw_top4": round(float(active_dropped[valid].mean()), 4),
        "correct_confirmed_slice_kept_only_by_pin": {
            "windows": int(rescued.sum()),
            "share_of_all_windows": round(float(rescued.sum() / valid.sum()), 4),
            "share_of_windows_with_correct_confirmed_slice": round(float(rescued.sum() / (active_correct & valid).sum()), 4),
            "episodes": len(episodes),
            "episode_length_windows": {
                "median": float(np.median(episodes)) if episodes else 0.0,
                "max": int(max(episodes)) if episodes else 0,
                "share_of_episodes_1_or_2_windows": round(float(np.mean(np.array(episodes) <= 2)), 4) if episodes else 0.0,
            },
            "share_within_60s_after_talk_boundary": round(float((since_boundary[rescued] <= 60).mean()), 4),
        },
        "windows_within_60s_after_talk_boundary_share": round(float((since_boundary[valid] <= 60).mean()), 4),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

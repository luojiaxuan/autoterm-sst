"""Unknown-domain fallback of the language-model router.

One topic is removed from the catalog and the language model gets an extra option,
"none of these glossaries fits". On the saved translations of the audited run, a
window falls back when "none" ranks first. A good router falls back during talks
whose topic has no slice and keeps a slice during the other talks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import llm_router_replay as lrr
from router_replay import DESCRIPTIONS, EN

NONE = "none"
NONE_TEXT = "None of these glossaries fits the talk."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--holdout", nargs="+", default=["nlp", "medicine"])
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = lrr.LetterScorer(args.llm, device, args.batch)

    recs = json.load(open(args.run, encoding="utf-8"))["records"]
    gold = np.array([str(r["expected_domain"]).lower() for r in recs])
    texts = [str(r.get("text") or "") for r in recs]
    ctx, nonempty = lrr.contexts_for(texts, args.w)
    ctx = [x if x.strip() else "(none)" for x in ctx]
    valid = np.arange(len(recs)) > 0

    result = {"model": args.llm, "holdouts": {}}
    for held in args.holdout:
        names = [n for n in EN if n != held] + [NONE]
        descriptions = {n: f"{DESCRIPTIONS[n]} {EN[n]}" for n in names if n != NONE}
        descriptions[NONE] = NONE_TEXT
        probs = lrr.carry_forward(scorer.score(ctx, names, descriptions), nonempty)
        np.save(args.out.with_suffix(f".{held}.npy"), probs)
        top = np.array(names)[probs.argmax(1)]
        unknown = valid & (gold == held)
        known = valid & (gold != held)
        row = {
            "fallback_when_topic_missing": round(float((top[unknown] == NONE).mean()), 4),
            "fallback_when_topic_present": round(float((top[known] == NONE).mean()), 4),
            "correct_slice_top1_when_present": round(float((top[known] == gold[known]).mean()), 4),
        }
        result["holdouts"][held] = row
        print(json.dumps({held: row}), flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Encode weak-supervision pseudo-contexts built from slice glossaries.

Gold entries harvested from the evaluation talks (sources acl_tagged_gt and
medicine_hard_llm_judge_manual) are excluded, so no evaluation term reaches the learned emission.
A pseudo-context is a random bag of n target-language translations from one slice, joined with the
Chinese enumeration comma; n is drawn to span the length of the w=8 routing contexts.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evidence"))
from router_replay import encode  # noqa: E402

EVAL_SOURCES = {"acl_tagged_gt", "medicine_hard_llm_judge_manual"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lang", default="zh")
    ap.add_argument("--per-class", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    texts, labels, kept = [], [], {}
    for path in sorted(args.glossary_dir.glob("*.json")):
        entries = [e for e in json.load(open(path, encoding="utf-8")) if e.get("source") not in EVAL_SOURCES]
        terms = [e["target_translations"][args.lang] for e in entries if e.get("target_translations", {}).get(args.lang)]
        kept[path.stem] = len(terms)
        for _ in range(args.per_class):
            texts.append("、".join(rng.sample(terms, rng.randint(3, 12))))
            labels.append(path.stem)
    emb = encode(model, tok, texts, device).astype(np.float32)
    np.savez_compressed(args.out, emb=emb, label=np.array(labels), text=np.array(texts))
    print(json.dumps({"kept_terms": kept, "contexts": len(texts)}))


if __name__ == "__main__":
    main()

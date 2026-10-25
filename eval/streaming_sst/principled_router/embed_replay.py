"""Encode the routing contexts and slice prototypes of an audited run once, for CPU-only router replays.

Encoding matches framework/agents/term_memory/context_similarity.py (BGE-M3 CLS, 128 tokens, left
truncation). The output npz holds per-window audio time, gold coarse domain, live-router state, and
context embeddings for each window size w; prototype embeddings are stored per slice and per text.
"""

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evidence"))
from router_replay import SUBFIELDS, VARIANTS, encode  # noqa: E402


def _slices(v):
    return v if isinstance(v, list) else ast.literal_eval(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", default="1,2,4,8")
    args = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    recs = json.load(open(args.run, encoding="utf-8"))["records"]
    texts = [str(r.get("text") or "") for r in recs]
    n = len(recs)
    out = {
        "cursor_s": np.array([float(r["cursor_s"]) for r in recs]),
        "gold": np.array([str(r["expected_domain"]).lower() for r in recs]),
        "live_active": np.array([str(r["active_domain"]).lower() for r in recs]),
        "live_slices": np.array([json.dumps(_slices(r["selected_slice_presets"])) for r in recs]),
        "text": np.array(texts),
    }
    for w in map(int, args.windows.split(",")):
        ctx = ["".join(texts[max(0, t - w):t]) for t in range(n)]
        out[f"nonempty_w{w}"] = np.array([bool(c.strip()) for c in ctx])
        out[f"ctx_w{w}"] = encode(model, tok, [c if c.strip() else " " for c in ctx], device).astype(np.float32)

    protos = {f"coarse:{d}": v for d, v in VARIANTS["original"].items()}
    protos.update({f"para:{d}": v for d, v in VARIANTS["paraphrase"].items()})
    for d, subs in SUBFIELDS.items():
        protos.update({f"fine:{d}:{s}": list(v) for s, v in subs.items()})
    names, flat, owner = [], [], []
    for k, v in protos.items():
        names.append(k)
        for t in v:
            flat.append(t)
            owner.append(len(names) - 1)
    out["proto_names"] = np.array(names)
    out["proto_owner"] = np.array(owner)
    out["proto_text_emb"] = encode(model, tok, flat, device).astype(np.float32)
    np.savez_compressed(args.out, **out)
    print(args.out, n, device)


if __name__ == "__main__":
    main()

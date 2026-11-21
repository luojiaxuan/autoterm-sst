"""Open-loop routing on German and Japanese partial translations.

RASST's paper runs translate the five ACL talks and the five ESO lectures into
German and Japanese with 1.92-second chunks. Each output unit carries the source
time at which it was emitted, so a talk's translation splits into per-chunk texts.
The ten talks are interleaved in the order of the audited Chinese run, and both
routers rank the ten topic slices from the last four chunks of target-language
text, exactly as for Chinese: cosine similarity of BGE-M3 embeddings, and the
letter probabilities of a small language model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import llm_router_replay as lrr
from router_replay import DESCRIPTIONS, EN, VARIANTS, centroids, encode, ranking_metrics

CHUNK_MS = 1920.0
LANG = {"de": "German", "ja": "Japanese"}


def chunk_texts(instance, joiner):
    units = instance["prediction"].split(" ") if joiner == " " else list(instance["prediction"])
    delays = json.loads(instance["delays"]) if isinstance(instance["delays"], str) else instance["delays"]
    assert len(units) == len(delays), (len(units), len(delays))
    # A unit emitted after reading k chunks carries delay k * 1920 ms; the last chunk may be shorter.
    n = max(1, int(np.ceil(max(delays) / CHUNK_MS)))
    chunks = [[] for _ in range(n)]
    for u, d in zip(units, delays):
        chunks[min(max(int(np.ceil(d / CHUNK_MS)) - 1, 0), n - 1)].append(u)
    return [joiner.join(c) for c in chunks]


def load(path):
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            out.append(json.loads(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(LANG), required=True)
    ap.add_argument("--acl", type=Path, required=True)
    ap.add_argument("--medicine", type=Path, required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--w", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    joiner = "" if args.lang == "ja" else " "

    acl = load(args.acl)
    med = load(args.medicine)
    texts, gold = [], []
    for i in range(max(len(acl), len(med))):
        for dom, talks in (("nlp", acl), ("medicine", med)):
            if i < len(talks):
                ct = chunk_texts(talks[i], joiner)
                texts.extend(ct)
                gold.extend([dom] * len(ct))
    names = list(EN)
    valid = np.ones(len(texts), dtype=bool)
    valid[0] = False
    ctx, nonempty = lrr.contexts_for(texts, args.w)

    tok = AutoTokenizer.from_pretrained(args.encoder)
    tok.truncation_side = "left"
    enc = AutoModel.from_pretrained(args.encoder).to(device).eval()
    cnames, c = centroids(enc, tok, VARIANTS["original"], device)
    emb = encode(enc, tok, [x if x.strip() else " " for x in ctx], device)
    for t in range(1, len(emb)):
        if not nonempty[t]:
            emb[t] = emb[t - 1]
    cos = emb @ c.T
    del enc
    torch.cuda.empty_cache()

    lrr.PROMPT = lrr.PROMPT.replace("into Chinese", f"into {LANG[args.lang]}").replace("Chinese output", f"{LANG[args.lang]} output")
    scorer = lrr.LetterScorer(args.llm, device, args.batch)
    descriptions = {n: f"{DESCRIPTIONS[n]} {EN[n]}" for n in names}
    llm = lrr.carry_forward(scorer.score([x if x.strip() else "(none)" for x in ctx], names, descriptions), nonempty)
    np.save(args.out.with_suffix(".cos.npy"), cos)
    np.save(args.out.with_suffix(".llm.npy"), llm)

    result = {
        "lang": args.lang,
        "windows": len(texts),
        "talks": {"acl": len(acl), "medicine": len(med)},
        "cosine": ranking_metrics(cos, cnames, gold, valid, ks=(1, 4)),
        "llm": ranking_metrics(llm, names, gold, valid, ks=(1, 4)),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

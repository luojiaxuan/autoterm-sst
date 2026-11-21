"""Open-loop replay of a language-model router on the audited ten-talk run.

The generated translation of the audited AutoTerm run is held fixed, as in
router_replay.py. For every decoder window, a small instruction-tuned language
model reads the previous w translated chunks and a lettered list of slice
descriptions, and the router scores each slice by the probability the model
assigns to its letter as the first answer token. Scores are averaged over the
listed order and its reverse, so that a slice's score does not depend on where it
appears in the prompt. Slices are ranked by this probability with no switch
guards, which makes the numbers comparable to the raw cosine ranking of
router_replay.py and rapid_switch_replay.py.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from router_replay import DESCRIPTIONS, EN, ranking_metrics

CHUNK_S = 1.92
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT = (
    "A live speech translator is translating a talk from English into Chinese. "
    "Below is its most recent Chinese output, followed by terminology glossaries, "
    "one per topic. Which glossary best matches the topic of the talk at this point?\n\n"
    "Recent translation:\n{context}\n\nGlossaries:\n{options}\n\n"
    "Answer with the letter of the best glossary only."
)


def option_block(names, descriptions):
    return "\n".join(f"{LETTERS[i]}. {descriptions[n]}" for i, n in enumerate(names))


class LetterScorer:
    """Probability of each option letter as the first answer token."""

    def __init__(self, model_id, device, batch):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device).eval()
        self.device = device
        self.batch = batch

    def letter_ids(self, k):
        ids = []
        for letter in LETTERS[:k]:
            pieces = self.tok.encode(letter, add_special_tokens=False)
            assert len(pieces) == 1, letter
            ids.append(pieces[0])
        return ids

    def score(self, contexts, names, descriptions):
        """Returns an array [len(contexts), len(names)] of probabilities, order-debiased."""
        k = len(names)
        ids = self.letter_ids(k)
        total = np.zeros((len(contexts), k))
        for order in (list(range(k)), list(range(k))[::-1]):
            listed = [names[i] for i in order]
            block = option_block(listed, descriptions)
            prompts = [
                self.tok.apply_chat_template(
                    [{"role": "user", "content": PROMPT.format(context=c, options=block)}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for c in contexts
            ]
            probs = []
            for i in range(0, len(prompts), self.batch):
                enc = self.tok(prompts[i:i + self.batch], return_tensors="pt", padding=True).to(self.device)
                with torch.inference_mode():
                    logits = self.model(**enc).logits[:, -1, :].float()
                p = torch.softmax(logits[:, ids], dim=-1).cpu().numpy()
                probs.append(p)
            p = np.concatenate(probs)
            # column j of p is the slice listed at position j; map back to the canonical order.
            for j, src in enumerate(order):
                total[:, src] += p[:, j]
        return total / 2


def contexts_for(texts, w):
    ctx = ["".join(texts[max(0, t - w):t]) for t in range(len(texts))]
    return ctx, np.array([bool(c.strip()) for c in ctx])


def carry_forward(scores, nonempty):
    # An empty context keeps the previous window's scores, as a live router would.
    for t in range(1, len(scores)):
        if not nonempty[t]:
            scores[t] = scores[t - 1]
    return scores


def rapid_streams(recs, spans, pieces):
    talks = []
    for s in spans:
        lo, hi = int(s["start_sample"]), int(s["end_sample"])
        talk = [str(r.get("text") or "") for r in recs if lo < int(r["cursor_samples"]) <= hi]
        talks.append((str(s["expected_domain"]).lower(), talk))
    streams = {}
    for piece_s in (15, 30, 60, 120, 300):
        n_chunks = max(1, round(piece_s / CHUNK_S))
        pools = {"nlp": [], "medicine": []}
        for dom, talk in talks:
            for i in range(0, len(talk) - n_chunks + 1, n_chunks):
                pools[dom].append(talk[i:i + n_chunks])
        texts, gold, pos = [], [], []
        for p in range(pieces):
            dom = "nlp" if p % 2 == 0 else "medicine"
            piece = pools[dom][(p // 2) * 7 % len(pools[dom])]
            texts.extend(piece)
            gold.extend([dom] * len(piece))
            pos.extend(range(len(piece)))
        streams[piece_s] = (texts, gold, np.array(pos))
    return streams


def switch_metrics(scores, names, gold, pos):
    order = np.argsort(-scores, axis=1)
    gidx = np.array([names.index(g) for g in gold])
    rank = np.argmax(order == gidx[:, None], axis=1)
    n_chunks = int(pos.max()) + 1
    # Same convention as rapid_switch_replay.py: the first segment has no preceding topic.
    valid = np.arange(len(gold)) >= n_chunks
    starts = np.flatnonzero((pos == 0) & valid)
    ends = [a + n_chunks for a in starts]

    def first(k):
        out, never = [], 0
        for a, b in zip(starts, ends):
            hit = np.flatnonzero(rank[a:b] < k)
            if len(hit):
                out.append(hit[0] * CHUNK_S)
            else:
                never += 1
        return (round(float(np.median(out)), 2) if out else None), never

    t1, n1 = first(1)
    t4, n4 = first(4)
    return {
        "cover@1": round(float((rank[valid] == 0).mean()), 4),
        "cover@4": round(float((rank[valid] < 4).mean()), 4),
        "median_seconds_until_gold_in_top1": t1,
        "pieces_where_gold_never_reaches_top1": n1,
        "median_seconds_until_gold_in_top4": t4,
        "pieces_where_gold_never_reaches_top4": n4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[4])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pieces", type=int, default=40)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = LetterScorer(args.model, device, args.batch)

    payload = json.load(open(args.run, encoding="utf-8"))
    recs = payload["records"]
    gold = [str(r["expected_domain"]).lower() for r in recs]
    texts = [str(r.get("text") or "") for r in recs]
    names = list(EN)
    descriptions = {n: f"{DESCRIPTIONS[n]} {EN[n]}" for n in names}
    valid = np.ones(len(recs), dtype=bool)
    valid[0] = False

    result = {"model": args.model, "windows": len(recs), "slices": names, "coarse": {}, "rapid_switch": {}}
    for w in args.windows:
        t0 = time.time()
        ctx, nonempty = contexts_for(texts, w)
        scores = carry_forward(scorer.score([c if c.strip() else "(none)" for c in ctx], names, descriptions), nonempty)
        np.save(args.out.with_suffix(f".w{w}.npy"), scores)
        row = ranking_metrics(scores, names, gold, valid, ks=(1, 2, 3, 4, 6))
        row["seconds"] = round(time.time() - t0, 1)
        result["coarse"][f"w={w}"] = row
        print(json.dumps({f"w={w}": row}), flush=True)

    for piece_s, (s_texts, s_gold, pos) in rapid_streams(recs, payload["block_spans"], args.pieces).items():
        ctx, nonempty = contexts_for(s_texts, 4)
        scores = carry_forward(scorer.score([c if c.strip() else "(none)" for c in ctx], names, descriptions), nonempty)
        result["rapid_switch"][f"piece={piece_s}s"] = switch_metrics(scores, names, s_gold, pos)
        print(json.dumps({f"piece={piece_s}s": result["rapid_switch"][f"piece={piece_s}s"]}), flush=True)

    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Split a domain glossary into fine topic slices with a small language model.

For each domain, the model first proposes K subfields from a random sample of the
domain's terms, with entries from the evaluation glossaries left out of the sample.
It then assigns every term of the domain to one subfield, scored as the
probability of the subfield's letter as the first answer token and averaged over
the listed order and its reverse. The subfield descriptions serve as the slices'
routing prototypes, so no prototype text is derived from evaluation terms.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from llm_router_replay import LETTERS, LetterScorer

EVAL_SOURCES = {"acl_tagged_gt", "medicine_hard_llm_judge_manual"}
TAXONOMY_PROMPT = (
    "Here are {n} terms from a terminology glossary about {domain}:\n{terms}\n\n"
    "Split this glossary into exactly {k} subfields that a speech translator could "
    "switch between during a talk. Subfields must not overlap, and together they must "
    "cover every term. Return a JSON list of {k} objects with the keys \"name\" "
    "(two to four English words), \"description_en\" (one English sentence listing the "
    "subfield's core topics and representative terms) and \"description_zh\" (its Chinese "
    "translation). Return only the JSON."
)
ASSIGN_PROMPT = (
    "Which subfield of {domain} does the glossary term \"{term}\" belong to?\n\n{options}\n\n"
    "Answer with the letter of the subfield only."
)


def propose_taxonomy(scorer, domain, terms, k, n):
    sample = random.sample(terms, min(n, len(terms)))
    prompt = TAXONOMY_PROMPT.format(n=len(sample), domain=domain, terms="\n".join(sample), k=k)
    text = scorer.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    enc = scorer.tok(text, return_tensors="pt").to(scorer.device)
    with torch.inference_mode():
        out = scorer.model.generate(**enc, max_new_tokens=2048, do_sample=False)
    reply = scorer.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    match = re.search(r"\[.*\]", reply, re.S)
    subfields = json.loads(match.group(0))
    assert len(subfields) == k, (domain, len(subfields))
    return subfields, sample, reply


def assign(scorer, domain, terms, subfields):
    k = len(subfields)
    ids = scorer.letter_ids(k)
    total = np.zeros((len(terms), k))
    for order in (list(range(k)), list(range(k))[::-1]):
        block = "\n".join(f"{LETTERS[j]}. {subfields[i]['name']}: {subfields[i]['description_en']}" for j, i in enumerate(order))
        prompts = [
            scorer.tok.apply_chat_template(
                [{"role": "user", "content": ASSIGN_PROMPT.format(domain=domain, term=t, options=block)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for t in terms
        ]
        probs = []
        for i in range(0, len(prompts), scorer.batch):
            enc = scorer.tok(prompts[i:i + scorer.batch], return_tensors="pt", padding=True).to(scorer.device)
            with torch.inference_mode():
                logits = scorer.model(**enc).logits[:, -1, :].float()
            probs.append(torch.softmax(logits[:, ids], dim=-1).cpu().numpy())
        p = np.concatenate(probs)
        for j, src in enumerate(order):
            total[:, src] += p[:, j]
    return total / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary", nargs="+", required=True, help="domain=path to a <domain>_core_10k.json")
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = LetterScorer(args.model, device, args.batch)
    names = {"nlp": "computer science and natural language processing", "medicine": "medicine and health care"}

    result = {"model": args.model, "k": args.k, "seed": args.seed, "domains": {}}
    for spec in args.glossary:
        domain, path = spec.split("=", 1)
        entries = json.load(open(path, encoding="utf-8"))
        terms = [e["term"] for e in entries]
        pool = [e["term"] for e in entries if e.get("source") not in EVAL_SOURCES]
        subfields, sample, reply = propose_taxonomy(scorer, names[domain], pool, args.k, args.sample)
        probs = assign(scorer, names[domain], terms, subfields)
        label = probs.argmax(1)
        result["domains"][domain] = {
            "subfields": subfields,
            "taxonomy_reply": reply,
            "taxonomy_sample": sample,
            "assignment": [
                {"term": e["term"], "source": e.get("source"), "subfield": int(label[i]), "prob": round(float(probs[i, label[i]]), 4)}
                for i, e in enumerate(entries)
            ],
            "sizes": np.bincount(label, minlength=args.k).tolist(),
        }
        print(json.dumps({domain: {"names": [s["name"] for s in subfields], "sizes": result["domains"][domain]["sizes"]}}), flush=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

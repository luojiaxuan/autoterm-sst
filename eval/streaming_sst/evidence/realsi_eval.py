"""TERM_ACC, BLEU and paired talk-level tests for the RealSI ten-domain streams.

A system is run on several stream orders; its metrics pool all orders. TERM_ACC hits come
from score_realsi_terms.py. BLEU resegments each talk's output against the RealSI
reference segments with mwerSegmenter at the character level and scores pooled sacreBLEU
with the zh tokenizer, as in the audited ten-talk pipeline. A paired comparison treats the
talk as the unit: the exact permutation test swaps two systems' outputs of a talk, in all
orders at once, over all 2^10 patterns, and the bootstrap resamples talks.
"""

import argparse
import importlib.util
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import sacrebleu


def load_scorer(path):
    spec = importlib.util.spec_from_file_location("bleu_scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def talk_outputs(payload):
    out = {}
    for span in payload["block_spans"]:
        lo, hi = int(span["start_sample"]), int(span["end_sample"])
        out[span["item_id"]] = "".join(str(r.get("text") or "") for r in payload["records"]
                                       if lo < int(r.get("cursor_samples") or 0) <= hi)
    return out


def bleu(pairs):
    hyps = [h for h, _ in pairs]
    refs = [r for _, r in pairs]
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", action="append", required=True, metavar="NAME=RUN_JSON[,RUN_JSON...]")
    ap.add_argument("--term-hits", type=Path, required=True, help="score_realsi_terms.py output over the same runs")
    ap.add_argument("--realsi-json", type=Path, required=True, help="folder of en2zh-*.json")
    ap.add_argument("--bleu-scorer", type=Path, required=True, help="score_merged_realistic_glossary.py")
    ap.add_argument("--mwer-segmenter", type=Path, required=True)
    ap.add_argument("--pair", action="append", default=[], metavar="A,B")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    scorer = load_scorer(args.bleu_scorer)
    refs = {p.stem: [s["trg_text"] for s in json.load(open(p, encoding="utf-8"))["segment"]]
            for p in sorted(args.realsi_json.glob("en2zh-*.json"))}
    talks = sorted(refs)

    hits = defaultdict(lambda: defaultdict(list))  # run name -> talk -> [hit]
    for row in json.load(open(args.term_hits, encoding="utf-8"))["occurrences"]:
        hits[row["run"]][row["talk"]].append(row["hit"])

    # per system and talk: lists over orders of (term hits, resegmented (hyp, ref) pairs)
    data = {}
    for spec in args.system:
        name, paths = spec.split("=", 1)
        per_talk = defaultdict(lambda: {"hits": [], "pairs": []})
        for k, path in enumerate(paths.split(",")):
            run_name = Path(path).stem
            outputs = talk_outputs(json.load(open(path, encoding="utf-8")))
            for talk in talks:
                segs = scorer.resegment_prediction(prediction=outputs[talk], references=refs[talk],
                                                   mwer_segmenter=args.mwer_segmenter, character_level=True)
                per_talk[talk]["pairs"] += list(zip(segs, refs[talk]))
                per_talk[talk]["hits"] += hits[run_name][talk]
        data[name] = per_talk

    def metrics(choice):  # choice: talk -> system name
        h = [x for t in talks for x in data[choice[t]][t]["hits"]]
        p = [x for t in talks for x in data[choice[t]][t]["pairs"]]
        return 100 * sum(h) / len(h), bleu(p)

    result = {"talks": talks, "systems": {}, "pairs": {}}
    for name in data:
        term, b = metrics({t: name for t in talks})
        result["systems"][name] = {"term_acc": round(term, 2), "bleu": round(b, 2),
                                   "per_talk_term_acc": {t: round(100 * np.mean(data[name][t]["hits"]), 1)
                                                         for t in talks}}
    rng = np.random.default_rng(args.seed)
    for spec in args.pair:
        a, b = spec.split(",")
        base = np.subtract(metrics({t: a for t in talks}), metrics({t: b for t in talks}))
        extreme = np.zeros(2)
        for signs in itertools.product((0, 1), repeat=len(talks)):
            ca = {t: (a if s == 0 else b) for t, s in zip(talks, signs)}
            cb = {t: (b if s == 0 else a) for t, s in zip(talks, signs)}
            d = np.subtract(metrics(ca), metrics(cb))
            extreme += np.abs(d) >= np.abs(base) - 1e-9
        boots = []
        for _ in range(args.n_boot):
            sample = rng.choice(talks, size=len(talks), replace=True)
            ha = [x for t in sample for x in data[a][t]["hits"]]
            hb = [x for t in sample for x in data[b][t]["hits"]]
            pa = [x for t in sample for x in data[a][t]["pairs"]]
            pb = [x for t in sample for x in data[b][t]["pairs"]]
            boots.append((100 * (np.mean(ha) - np.mean(hb)), bleu(pa) - bleu(pb)))
        boots = np.array(boots)
        result["pairs"][f"{a}-{b}"] = {
            metric: {"diff": round(float(base[i]), 2), "permutation_p": round(float(extreme[i] / 2 ** len(talks)), 4),
                     "bootstrap_ci95": [round(float(x), 2) for x in np.percentile(boots[:, i], [2.5, 97.5])]}
            for i, metric in enumerate(("term_acc", "bleu"))}
    result["n_boot"], result["seed"] = args.n_boot, args.seed
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"systems": {k: {x: v[x] for x in ("term_acc", "bleu")} for k, v in result["systems"].items()},
                      "pairs": result["pairs"]}, indent=1))


if __name__ == "__main__":
    main()

"""COMET on the same resegmented sentences as BLEU, with the talk-level paired tests used for BLEU.

Each --run gives a system name and its segments.jsonl (one row per reference sentence with source, prediction,
reference and the talk's wav). The model scores every sentence; a system's score is the mean over all sentences,
also reported per corpus (ACL talks versus ESO talks, told apart by the wav name). For each --pair A,B the
difference A minus B is tested with the exact talk permutation (each of the 2^talks ways of swapping the two
systems' outputs within talks) and a talk bootstrap stratified by corpus, as paired_significance.py does for BLEU.

With --term-glossary for both corpora, every statistic is also computed on two sentence subsets: term-free
sentences, whose reference holds no target term of its corpus glossary (the definition of term_free_quality.py),
and the term-bearing rest. --scores-out writes every sentence's score.

usage: comet_quality.py --model Unbabel/wmt22-comet-da --run name=segments.jsonl ... --pair a,b ...
       [--term-glossary acl=<json> --term-glossary eso=<json> --scorer-code <dir>] [--scores-out s.jsonl] --out out.json
"""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

from comet import download_model, load_from_checkpoint


def corpus_of(wav):
    return "acl" if wav.startswith("2022.acl") else "eso"


def pooled(scores_by_talk, talks):
    values = [score for talk in talks for score in scores_by_talk[talk]]
    return sum(values) / len(values)


def summarize(scores, talks, pairs, n_boot, seed):
    """Means overall and per corpus, and the paired tests, for per-talk score lists of every system."""
    corpora = {corpus: [t for t in talks if corpus_of(t) == corpus] for corpus in ("acl", "eso")}
    result = {"sentences": {name: sum(len(v) for v in by_talk.values()) for name, by_talk in scores.items()},
              "systems": {}, "pairs": {}}
    for name, by_talk in scores.items():
        result["systems"][name] = {"all": pooled(by_talk, talks),
                                   **{corpus: pooled(by_talk, ts) for corpus, ts in corpora.items()}}
    rng = random.Random(seed)
    for a, b in pairs:
        diff = result["systems"][a]["all"] - result["systems"][b]["all"]
        extreme = 0
        for swaps in itertools.product((False, True), repeat=len(talks)):
            left = {t: (scores[b] if s else scores[a])[t] for t, s in zip(talks, swaps)}
            right = {t: (scores[a] if s else scores[b])[t] for t, s in zip(talks, swaps)}
            extreme += abs(pooled(left, talks) - pooled(right, talks)) >= abs(diff) - 1e-12
        boot = []
        for _ in range(n_boot):
            sample = [rng.choice(ts) for ts in corpora.values() for _ in ts]
            boot.append(pooled(scores[a], sample) - pooled(scores[b], sample))
        boot.sort()
        result["pairs"][f"{a}_vs_{b}"] = {
            "diff": diff, "talk_permutation_p": extreme / 2 ** len(talks),
            "stratified_talk_bootstrap_ci95": [boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot) - 1]]}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--pair", action="append", default=[])
    ap.add_argument("--term-glossary", action="append", default=[], help="corpus=path, for acl and eso")
    ap.add_argument("--scorer-code", type=Path, help="tree that holds eval/streaming_sst/score_terms.py")
    ap.add_argument("--scores-out", type=Path)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    terms = {}
    if args.term_glossary:
        sys.path.insert(0, str(args.scorer_code))
        from eval.streaming_sst.score_terms import load_target_terms_for_masking

        terms = {corpus: load_target_terms_for_masking(path, "zh")
                 for corpus, path in (item.split("=", 1) for item in args.term_glossary)}
        if set(terms) != {"acl", "eso"}:
            raise ValueError("--term-glossary needs both acl and eso")

    model = load_from_checkpoint(download_model(args.model))
    rows_by_system = {}
    for item in args.run:
        name, path = item.split("=", 1)
        rows_by_system[name] = [json.loads(line) for line in open(path)]
    references = {tuple((r["wav"], r["sentence_index_in_talk"], r["reference"]) for r in rows)
                  for rows in rows_by_system.values()}
    assert len(references) == 1, "systems were scored on different reference sentences"

    scored = {}
    for name, rows in rows_by_system.items():
        data = [{"src": r["source"], "mt": r["prediction"], "ref": r["reference"]} for r in rows]
        output = model.predict(data, batch_size=args.batch_size, gpus=1, progress_bar=False)
        scored[name] = [(row, float(score)) for row, score in zip(rows, output.scores)]
    if args.scores_out:
        with args.scores_out.open("w") as out:
            for name, items in scored.items():
                for row, score in items:
                    out.write(json.dumps({"system": name, "wav": row["wav"], "sentence_index_in_talk":
                                          row["sentence_index_in_talk"], "comet": score}) + "\n")

    def free(row):
        corpus = corpus_of(row["wav"])
        return not any(t and t in row["reference"] for t in terms[corpus])

    subsets = {"all": lambda row: True}
    if terms:
        subsets.update(term_free=free, term_bearing=lambda row: not free(row))
    talks = sorted({row["wav"] for row, _ in next(iter(scored.values()))})
    pairs = [tuple(pair.split(",")) for pair in args.pair]
    result = {"model": args.model, "talks": talks, "subsets": {}}
    for subset, keep in subsets.items():
        scores = {}
        for name, items in scored.items():
            by_talk = {t: [] for t in talks}
            for row, score in items:
                if keep(row):
                    by_talk[row["wav"]].append(score)
            scores[name] = by_talk
        result["subsets"][subset] = summarize(scores, talks, pairs, args.n_boot, args.seed)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for subset, value in result["subsets"].items():
        print(f"[{subset}] sentences {value['sentences']}")
        for name, s in value["systems"].items():
            print(f"  {name:12s} all {s['all']:.4f} acl {s['acl']:.4f} eso {s['eso']:.4f}")
        for key, p in value["pairs"].items():
            print(f"  {key:28s} diff {p['diff']:+.4f} p {p['talk_permutation_p']:.3f} "
                  f"ci {p['stratified_talk_bootstrap_ci95']}")


if __name__ == "__main__":
    main()

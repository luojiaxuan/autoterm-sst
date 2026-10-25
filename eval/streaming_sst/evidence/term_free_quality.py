"""Translation quality on sentences whose reference contains no glossary target string.

The subset is defined from the reference alone, so it does not condition on either system's
term outcome and needs no masking. Policies are compared with pooled sacreBLEU on that
subset; inference uses an exact talk-level label-swap permutation test and a
domain-stratified talk-cluster bootstrap, as in paired_significance.py.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_significance import Corpus  # noqa: E402



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--gloss-dir", type=Path, required=True)
    ap.add_argument("--run-data", action="append", required=True, metavar="NAME=SEGMENTS_JSONL,BUNDLE_MANIFEST")
    ap.add_argument("--pair", action="append", required=True, metavar="A,B")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    from eval.streaming_sst.score_terms import load_target_terms_for_masking

    gloss = args.gloss_dir
    run_data = {name: paths.split(",") for name, paths in (spec.split("=", 1) for spec in args.run_data)}
    PAIRS = [tuple(spec.split(",")) for spec in args.pair]
    terms = {
        "acl": load_target_terms_for_masking(str(gloss / "acl6060_tagged_gt_raw_min_norm2.json"), "zh"),
        "medicine": load_target_terms_for_masking(str(gloss / "hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json"), "zh"),
    }
    rng = np.random.default_rng(20261024)
    runs = sorted({r for p in PAIRS for r in p})
    rows, keys, block_of_wav, domain_of_block = {}, None, {}, {}
    for r in runs:
        manifest = json.load(open(run_data[r][1], encoding="utf-8"))
        for b in manifest["blocks"]:
            block_of_wav[b["wav"]] = int(b["block_index"])
            domain_of_block[int(b["block_index"])] = b["corpus"]
        rows[r] = [json.loads(line) for line in open(run_data[r][0], encoding="utf-8") if line.strip()]
        row_keys = [(x["wav"], x["sentence_index_in_talk"], x["reference"]) for x in rows[r]]
        if keys is None:
            keys = row_keys
        assert row_keys == keys, f"{r}: sentence order differs"

    term_free = np.array([
        not any(t and t in ref for t in terms[domain_of_block[block_of_wav[wav]]])
        for wav, _, ref in keys
    ])
    subset = np.flatnonzero(term_free)
    sent_talk = np.array([block_of_wav[k[0]] for k in keys])
    talks = sorted(domain_of_block)
    strata = {d: [t for t in talks if domain_of_block[t] == d] for d in ("acl", "medicine")}
    members = {t: subset[sent_talk[subset] == t] for t in talks}
    corp = {r: Corpus([x["prediction"] for x in rows[r]], [x["reference"] for x in rows[r]]) for r in runs}

    def score(run, talk_list):
        return corp[run].score(np.concatenate([members[t] for t in talk_list]))

    draws = [np.concatenate([rng.choice(strata["acl"], size=5), rng.choice(strata["medicine"], size=5)]) for _ in range(args.n_boot)]
    result = {"term_free_sentences": int(len(subset)), "sentences": len(keys), "by_domain": {d: int(sum(len(members[t]) for t in strata[d])) for d in strata}}
    for a, b in PAIRS:
        point = score(a, talks) - score(b, talks)
        boot = np.array([score(a, d) - score(b, d) for d in draws])
        assert np.isfinite(boot).all()
        perm = []
        for flip in itertools.product([False, True], repeat=len(talks)):
            sa, sb = corp[a].stats.copy(), corp[b].stats.copy()
            for t, f in zip(talks, flip):
                if f:
                    m = members[t]
                    sa[m], sb[m] = corp[b].stats[m], corp[a].stats[m]
            sa_sum, sb_sum = sa[subset].sum(0), sb[subset].sum(0)
            perm.append(corp[a].bleu._compute_score_from_stats(list(sa_sum)).score - corp[a].bleu._compute_score_from_stats(list(sb_sum)).score)
        perm = np.array(perm)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        result[f"{a}_vs_{b}"] = {
            "bleu_diff": point,
            "stratified_talk_bootstrap_ci95": [float(lo), float(hi)],
            "talk_permutation_p": float((np.abs(perm) >= abs(point) - 1e-12).mean()),
            "per_domain_diff": {d: score(a, strata[d]) - score(b, strata[d]) for d in strata},
        }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

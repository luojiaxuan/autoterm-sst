"""Paired significance for the ten-talk table: TERM_ACC, BLEU, MT-BLEU.

Primary inference treats talks as the sampling unit:
- exact talk-level label-swap permutation test (all 2^10 sign patterns, pooled metric
  recomputed for each pattern), Holm-adjusted across the reported contrasts;
- domain-stratified talk-cluster bootstrap (5 ACL + 5 ESO talks per draw) for 95% CIs.
Occurrence-level McNemar and sentence-level bootstrap are kept as sensitivity analyses.
BLEU is always computed from pooled sacreBLEU sufficient statistics with sacreBLEU's own
scoring (exp smoothing), so every resample is a valid corpus BLEU.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

RUNS = ["known", "autoterm", "merged100k", "merged1m"]
PAIRS = [("autoterm", "known"), ("autoterm", "merged100k"), ("autoterm", "merged1m"), ("known", "merged1m")]
METRICS = ("term_acc", "bleu", "mt_bleu")


def holm(pvals: dict) -> dict:
    order = sorted(pvals, key=pvals.get)
    m, running, out = len(order), 0.0, {}
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[key]))
        out[key] = running
    return out


class Corpus:
    """Per-sentence BLEU statistics for one policy, scored with sacreBLEU semantics."""

    def __init__(self, hyps, refs):
        from sacrebleu.metrics import BLEU

        self.bleu = BLEU(tokenize="zh")
        self.stats = np.array(self.bleu._extract_corpus_statistics(hyps, [refs]), dtype=np.float64)
        self.full = self.bleu.corpus_score(hyps, [refs]).score
        assert abs(self.score(np.arange(len(hyps))) - self.full) < 1e-9

    def score(self, idx) -> float:
        return float(self.bleu._compute_score_from_stats(list(self.stats[idx].sum(axis=0))).score)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--staging", type=Path, required=True)
    ap.add_argument("--occurrences", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    occ = [json.loads(line) for line in open(args.occurrences, encoding="utf-8")]
    hit = {r: np.array([o[f"hit_{r}"] for o in occ], dtype=np.float64) for r in RUNS}
    occ_talk = np.array([o["block_index"] for o in occ])

    sys.path.insert(0, str(args.repo))
    from eval.streaming_sst.score_terms import compile_term_mask_patterns, load_target_terms_for_masking, mask_target_terms

    gloss = args.staging / "gold/RASST/data/glossaries"
    patterns = {
        "acl": compile_term_mask_patterns(load_target_terms_for_masking(str(gloss / "acl6060_tagged_gt_raw_min_norm2.json"), "zh")),
        "medicine": compile_term_mask_patterns(load_target_terms_for_masking(str(gloss / "hard_medicine_glossary_raw_llm_judge_manual_zh215_unique212.json"), "zh")),
    }
    corpora, keys, block_of_wav, domain_of_block = {}, None, {}, {}
    for r in RUNS:
        manifest = json.load(open(args.staging / "bundles" / r / "manifest.json", encoding="utf-8"))
        for b in manifest["blocks"]:
            block_of_wav[b["wav"]] = int(b["block_index"])
            domain_of_block[int(b["block_index"])] = b["corpus"]
        rows = [json.loads(line) for line in open(args.staging / "scores" / r / "segments.jsonl", encoding="utf-8") if line.strip()]
        row_keys = [(row["wav"], row["sentence_index_in_talk"], row["reference"]) for row in rows]
        if keys is None:
            keys = row_keys
        assert row_keys == keys, f"{r}: sentence order differs"
        hyps = [row["prediction"] for row in rows]
        refs = [row["reference"] for row in rows]
        dom = [domain_of_block[block_of_wav[row["wav"]]] for row in rows]
        corpora[("bleu", r)] = Corpus(hyps, refs)
        corpora[("mt_bleu", r)] = Corpus(
            [mask_target_terms(h, patterns[d])[0] for h, d in zip(hyps, dom)],
            [mask_target_terms(t, patterns[d])[0] for t, d in zip(refs, dom)],
        )
    sent_talk = np.array([block_of_wav[k[0]] for k in keys])
    talks = sorted(domain_of_block)
    strata = {d: [t for t in talks if domain_of_block[t] == d] for d in ("acl", "medicine")}
    occ_members = {t: np.flatnonzero(occ_talk == t) for t in talks}
    sent_members = {t: np.flatnonzero(sent_talk == t) for t in talks}

    def metric_value(metric, run, talk_list):
        if metric == "term_acc":
            idx = np.concatenate([occ_members[t] for t in talk_list])
            return 100.0 * hit[run][idx].mean()
        idx = np.concatenate([sent_members[t] for t in talk_list])
        return corpora[(metric, run)].score(idx)

    def swapped_value(metric, a, b, flip):
        # note (luojiaxuan): a talk with flip=True contributes B's output under A's label and vice versa.
        if metric == "term_acc":
            ha, hb = hit[a].copy(), hit[b].copy()
            for t, f in zip(talks, flip):
                if f:
                    m = occ_members[t]
                    ha[m], hb[m] = hit[b][m], hit[a][m]
            return 100.0 * (ha.mean() - hb.mean())
        ca, cb = corpora[(metric, a)], corpora[(metric, b)]
        sa, sb = ca.stats.copy(), cb.stats.copy()
        for t, f in zip(talks, flip):
            if f:
                m = sent_members[t]
                sa[m], sb[m] = cb.stats[m], ca.stats[m]
        return float(ca.bleu._compute_score_from_stats(list(sa.sum(0))).score - ca.bleu._compute_score_from_stats(list(sb.sum(0))).score)

    boot_draws = [
        np.concatenate([rng.choice(strata["acl"], size=len(strata["acl"])), rng.choice(strata["medicine"], size=len(strata["medicine"]))])
        for _ in range(args.n_boot)
    ]
    patterns_all = list(itertools.product([False, True], repeat=len(talks)))

    result = {"n_boot": args.n_boot, "seed": args.seed, "talks": talks, "contrasts": {}}
    perm_p = {}
    for a, b in PAIRS:
        for metric in METRICS:
            point = metric_value(metric, a, talks) - metric_value(metric, b, talks)
            boot = np.array([metric_value(metric, a, d) - metric_value(metric, b, d) for d in boot_draws])
            assert np.isfinite(boot).all()
            perm = np.array([swapped_value(metric, a, b, f) for f in patterns_all])
            p_perm = float((np.abs(perm) >= abs(point) - 1e-12).mean())
            lo, hi = np.percentile(boot, [2.5, 97.5])
            entry = {
                "diff": point,
                "stratified_talk_bootstrap_ci95": [float(lo), float(hi)],
                "talk_permutation_p": p_perm,
                "per_domain_diff": {
                    d: metric_value(metric, a, strata[d]) - metric_value(metric, b, strata[d]) for d in strata
                },
                "leave_one_talk_out_range": [
                    float(min(metric_value(metric, a, [x for x in talks if x != t]) - metric_value(metric, b, [x for x in talks if x != t]) for t in talks)),
                    float(max(metric_value(metric, a, [x for x in talks if x != t]) - metric_value(metric, b, [x for x in talks if x != t]) for t in talks)),
                ],
            }
            if metric == "term_acc":
                d = hit[a] - hit[b]
                a_only, b_only = int((d > 0).sum()), int((d < 0).sum())
                entry["sensitivity_occurrence_mcnemar"] = {"a_only": a_only, "b_only": b_only, "p": float(binomtest(a_only, a_only + b_only, 0.5).pvalue)}
            result["contrasts"][f"{a}_vs_{b}:{metric}"] = entry
            perm_p[f"{a}_vs_{b}:{metric}"] = p_perm
    for key, p in holm(perm_p).items():
        result["contrasts"][key]["talk_permutation_p_holm"] = p
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for key, v in result["contrasts"].items():
        print(f"{key:32s} diff={v['diff']:+.3f} CI=[{v['stratified_talk_bootstrap_ci95'][0]:+.3f},{v['stratified_talk_bootstrap_ci95'][1]:+.3f}] "
              f"perm_p={v['talk_permutation_p']:.4g} holm={v['talk_permutation_p_holm']:.4g} "
              f"acl={v['per_domain_diff']['acl']:+.2f} med={v['per_domain_diff']['medicine']:+.2f} "
              f"loto=[{v['leave_one_talk_out_range'][0]:+.2f},{v['leave_one_talk_out_range'][1]:+.2f}]")


if __name__ == "__main__":
    main()

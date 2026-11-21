"""Per-occurrence TERM_ACC hits for the ten-talk RAG runs.

Reuses gold construction and matching from eval/streaming_sst/score_time_aligned_terms.py
and records, for every deduplicated gold occurrence, whether each run hit it.
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--gold-root", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_JSON")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.repo))
    import eval.streaming_sst.score_time_aligned_terms as st

    gold_root = args.gold_root
    ns = argparse.Namespace(
        mfa_root=str(args.repo / "eval/streaming_sst/mfa_alignments"),
        acl_root=str(gold_root / "rasst_eval/acl6060_zh_segments"),
        acl_technical_gold=str(args.repo / "eval/streaming_sst/acl_gold_technical.json"),
        acl_raw_glossary=str(gold_root / "RASST/data/glossaries/acl6060_tagged_gt_raw_min_norm2.json"),
        medicine_oracle_dir=str(gold_root / "RASST/data/main_result/inputs/medicine_zh"),
        target_lang="zh",
        post_s=30.0,
    )
    run_paths = dict(spec.split("=", 1) for spec in args.run)
    runs = list(run_paths)
    payloads = {r: json.load(open(run_paths[r], encoding="utf-8")) for r in runs}
    gold = st.deduplicate_alias_occurrences(st.build_timed_gold(payloads[runs[0]], ns)["raw_plus_medicine"])
    spans = [(s["start_sample"], s["end_sample"]) for s in payloads[runs[0]]["block_spans"]]
    for r in runs[1:]:
        assert [(s["start_sample"], s["end_sample"]) for s in payloads[r]["block_spans"]] == spans, f"{r}: block spans differ"

    groups = {}
    for idx, occ in enumerate(gold):
        groups.setdefault((occ.block_index, occ.term, tuple(occ.variants)), []).append(idx)

    hits = {r: [0] * len(gold) for r in runs}
    for r in runs:
        cache = {}
        for (block_index, term, variants), idxs in groups.items():
            if block_index not in cache:
                cache[block_index] = st.build_output_index(payloads[r], block_index)
            times = st.variant_match_times(cache[block_index], term, variants, "zh")
            idxs = sorted(idxs, key=lambda i: gold[i].t_start)
            used = [False] * len(times)
            for i in idxs:
                occ = gold[i]
                lo, hi = occ.t_start - st.PRE_S, occ.t_end + st.POST_S
                for j, mt in enumerate(times):
                    if used[j] or mt < lo:
                        continue
                    if mt > hi:
                        break
                    used[j] = True
                    hits[r][i] = 1
                    break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, occ in enumerate(gold):
            f.write(json.dumps({
                "idx": i, "domain": occ.domain, "block_index": occ.block_index, "term": occ.term,
                "variants": list(occ.variants), "t_start": occ.t_start, "t_end": occ.t_end,
                **{f"hit_{r}": hits[r][i] for r in runs},
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"gold": len(gold), **{r: sum(hits[r]) for r in runs}}))


if __name__ == "__main__":
    main()

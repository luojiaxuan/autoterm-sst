"""Time-aligned TERM_ACC on RealSI streams, with the matching rules of score_time_aligned_terms.py.

A gold occurrence is an annotated term of a RealSI sub-utterance (annotate_realsi_terms.py).
It is hit when an acceptable Chinese rendering appears in output emitted between 2 s before
the sub-utterance starts and 30 s after it ends; each output appearance satisfies at most one
occurrence, in time order. Acceptable renderings of a source term are every rendering the
talk's reference gives it, each also without a parenthetical gloss ("即时检验（POCT）" also
accepts "即时检验"). Because the reference endorses each rendering, a Latin-script one such as
"Google" counts when the output keeps it, matched on word boundaries, and the appearances of all
renderings are pooled, dropping a match that lies inside a longer one ("律师" inside "辩护律师").
Talks are located in the stream by the item ids of the run's block spans.
"""

import argparse
import json
import re
import sys
from pathlib import Path

PAREN = re.compile(r"[（(][^（）()]*[）)]")
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def match_times(index, variants):
    spans = []
    for v in variants:
        if CJK.search(v):
            pattern = re.escape(v)
        else:
            pattern = r"(?<![A-Za-z0-9])" + re.escape(v) + r"(?![A-Za-z0-9])"
        spans += [(m.start(), m.end()) for m in re.finditer(pattern, index.text, re.IGNORECASE)]
    kept = []
    for a, b in sorted(spans, key=lambda x: (x[0], -x[1])):
        if kept and a >= kept[-1][0] and b <= kept[-1][1]:
            continue
        kept.append((a, b))
    return sorted(index.time_at(a) for a, _ in kept)


def variants_by_source(talk):
    out = {}
    for t in talk["terms"]:
        key = " ".join(t["src"].casefold().split())
        for v in (t["trg"], PAREN.sub("", t["trg"]).strip()):
            if v:
                out.setdefault(key, set()).add(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--terms", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_JSON")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo))
    import eval.streaming_sst.score_time_aligned_terms as st

    talks = {t["talk"]: t for t in json.load(open(args.terms, encoding="utf-8"))["talks"]}
    rows, summary = [], {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        payload = json.load(open(path, encoding="utf-8"))
        hits = total = 0
        per_talk = {}
        for span in payload["block_spans"]:
            talk = talks[span["item_id"]]
            start_s = int(span["start_sample"]) / st.TARGET_SAMPLE_RATE
            index = st.build_output_index(payload, int(span["block_index"]))
            variants = variants_by_source(talk)
            groups = {}
            for i, t in enumerate(talk["terms"]):
                groups.setdefault(" ".join(t["src"].casefold().split()), []).append(i)
            talk_hits = [0] * len(talk["terms"])
            for key, idxs in groups.items():
                times = match_times(index, sorted(variants[key]))
                used = [False] * len(times)
                for i in sorted(idxs, key=lambda j: talk["terms"][j]["start_ms"]):
                    t = talk["terms"][i]
                    lo = start_s + t["start_ms"] / 1000 - st.PRE_S
                    hi = start_s + t["end_ms"] / 1000 + st.POST_S
                    for j, mt in enumerate(times):
                        if used[j] or mt < lo:
                            continue
                        if mt > hi:
                            break
                        used[j] = True
                        talk_hits[i] = 1
                        break
            for i, t in enumerate(talk["terms"]):
                rows.append({"run": name, "talk": talk["talk"], "domain": talk["domain"], "idx": i,
                             "src": t["src"], "trg": t["trg"], "hit": talk_hits[i]})
            per_talk[talk["talk"]] = [sum(talk_hits), len(talk_hits)]
            hits += sum(talk_hits)
            total += len(talk_hits)
        summary[name] = {"term_acc": round(100 * hits / total, 2), "hits": hits, "occurrences": total,
                         "per_talk": per_talk}
    args.out.write_text(json.dumps({"summary": summary, "occurrences": rows}, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: {x: v[x] for x in ("term_acc", "hits", "occurrences")} for k, v in summary.items()}))


if __name__ == "__main__":
    main()

"""Keyword-channel multi-domain falsification.

Scores the REAL generated-target output from an existing 10-talk run against
every registered domain using the production topic_keyword_scores, in sliding
windows, and reports how often distractor domains threaten the true domain.
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
# import the taxonomy module directly to avoid the framework package __init__
import importlib.util
REPO = "/mnt/taurus/home/jiaxuanluo/rasst-demo"
spec = importlib.util.spec_from_file_location(
    "domain_taxonomy", f"{REPO}/framework/agents/term_memory/domain_taxonomy.py")
dt = importlib.util.module_from_spec(spec); sys.modules["domain_taxonomy"] = dt; spec.loader.exec_module(dt)

def block_outputs(payload):
    spans = payload.get("block_spans") or []
    out = {int(s["block_index"]): [] for s in spans}
    for r in payload.get("records") or []:
        cur = int(r.get("cursor_samples") or 0)
        for s in spans:
            if int(s["start_sample"]) < cur <= int(s["end_sample"]):
                out[int(s["block_index"])].append(str(r.get("text") or r.get("text_preview") or ""))
                break
    return {i: "".join(p) for i, p in out.items()}

def windows(text, n_chars=160, stride=80):
    text = re.sub(r"\s+", "", text)
    if not text: return []
    return [text[i:i+n_chars] for i in range(0, max(1, len(text)-n_chars+1), stride)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--candidate-domains", default="")  # empty = all registered
    args = ap.parse_args()
    payload = json.load(open(args.run))
    spans = {int(s["block_index"]): s for s in payload.get("block_spans") or []}
    blocks = payload.get("blocks") or []
    outs = block_outputs(payload)

    all_domains = list(dt.DOMAIN_TOPIC_KEYWORDS.keys())
    cand = [d.strip() for d in args.candidate_domains.split(",") if d.strip()] or all_domains
    print(f"registered domains: {all_domains}")
    print(f"candidate set ({len(cand)}): {cand}\n")

    # per true-domain aggregation
    agg = {}   # true_domain -> {windows, top1_correct, distractor_top1, distractor_hits:{d:count}}
    for bi, block in enumerate(blocks, start=1):
        corpus = block.get("corpus")
        true = "nlp" if corpus == "acl" else ("medicine" if corpus == "medicine" else None)
        if true is None: continue
        a = agg.setdefault(true, {"win":0,"top1_ok":0,"distr_top1":0,"distr":{}})
        for w in windows(outs.get(bi, "")):
            scores, hits = dt.topic_keyword_scores(w)
            # restrict to candidate set
            cs = {d: scores.get(d, 0.0) for d in cand}
            if all(v == 0 for v in cs.values()):
                continue  # no signal window (router would keep current slice)
            a["win"] += 1
            ranked = sorted(cs.items(), key=lambda kv: kv[1], reverse=True)
            top_d, top_v = ranked[0]
            if top_d == true:
                a["top1_ok"] += 1
            else:
                a["distr_top1"] += 1
            # any distractor (non-true, non-general) firing at all
            for d, v in cs.items():
                if v > 0 and d != true and d != "general":
                    a["distr"][d] = a["distr"].get(d, 0) + 1
    for true, a in agg.items():
        w = a["win"] or 1
        print(f"=== true domain: {true}  ({a['win']} signal windows) ===")
        print(f"  top-1 correct:      {a['top1_ok']}/{a['win']} = {a['top1_ok']/w:.3f}")
        print(f"  distractor top-1:   {a['distr_top1']}/{a['win']} = {a['distr_top1']/w:.3f}")
        top_distr = sorted(a["distr"].items(), key=lambda kv: kv[1], reverse=True)[:6]
        print(f"  distractor firings: {top_distr}")
        print()

if __name__ == "__main__":
    main()

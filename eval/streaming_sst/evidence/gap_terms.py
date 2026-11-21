"""Gold term occurrences that fall in decoder windows whose working set lacks the correct slice."""
import bisect, json, sys
staging, occ_path = sys.argv[1], sys.argv[2]
p = json.load(open(f"{staging}/runs/autoterm.json", encoding="utf-8"))
recs = p["records"]
cur = [float(r["cursor_s"]) for r in recs]
def dom(x):
    x = str(x).lower()
    return "nlp" if x.startswith("nlp_") else "medicine" if x.startswith("medicine_") else "other"
absent = [str(r.get("expected_domain") or "").lower() not in [dom(s) for s in (r.get("selected_slice_presets") or [])] for r in recs]
occ = [json.loads(l) for l in open(occ_path, encoding="utf-8")]
inside = []
for o in occ:
    i = bisect.bisect_left(cur, float(o["t_start"]))
    i = min(i, len(recs) - 1)
    if absent[i]:
        inside.append(o)
n = len(inside)
print(json.dumps({
    "occurrences": len(occ), "in_gap_windows": n,
    "autoterm_hits_in_gap": sum(o["hit_autoterm"] for o in inside),
    "known_hits_in_gap": sum(o["hit_known"] for o in inside),
    "merged1m_hits_in_gap": sum(o["hit_merged1m"] for o in inside),
    "by_domain": {d: sum(1 for o in inside if o["domain"] == d) for d in ("nlp", "medicine")},
}))

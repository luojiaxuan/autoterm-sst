"""Carve domain-representative slices out of the 1M wiki pool using the
production DOMAIN_KEYWORDS, scoring term+description. No crawl, keeps zh."""
import importlib.util, json, sys, re
RT = "/mnt/taurus/data2/jiaxuanluo/rasst-demo/runtime/term_memory"
REPO = "/mnt/taurus/home/jiaxuanluo/rasst-demo"
spec = importlib.util.spec_from_file_location("domain_taxonomy", f"{REPO}/framework/agents/term_memory/domain_taxonomy.py")
dt = importlib.util.module_from_spec(spec); sys.modules["domain_taxonomy"] = dt; spec.loader.exec_module(dt)

pool = json.load(open(f"{RT}/glossaries/wiki_general_zh_1m.json"))
print("pool:", len(pool))

TARGET = 100000
for domain in ("finance", "legal"):
    kws = [k.lower() for k in dt.DOMAIN_KEYWORDS.get(domain, ())]
    pats = [re.compile(r"(?<![a-z])" + re.escape(k) + r"(?![a-z])") for k in kws]
    scored = []
    for e in pool:
        blob = (str(e.get("term","")) + " " + str(e.get("short_description",""))).lower()
        if not blob.strip():
            continue
        hits = sum(1 for p in pats if p.search(blob))
        if hits > 0:
            scored.append((hits, e))
    scored.sort(key=lambda x: x[0], reverse=True)
    keep = [e for _, e in scored[:TARGET]]
    for e in keep:
        e["source"] = f"wiki_{domain}_carved"
    out = f"{RT}/glossaries/{domain}_wiki_100k.json"
    json.dump(keep, open(out, "w"), ensure_ascii=False)
    print(f"{domain}: {len(scored)} matched -> kept {len(keep)}  (min hits in kept: {scored[min(len(scored),TARGET)-1][0] if scored else 0})")
    print("  sample:", [e["term"] for e in keep[:6]])

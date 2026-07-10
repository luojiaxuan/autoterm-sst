import json, sys, time, urllib.parse, urllib.request
API="https://www.wikidata.org/w/api.php"
UA="AutoTerm-SST/1.0 (academic; mailto:jluo50@jhu.edu)"
LANG="ja"
def fetch(ids):
    url=API+"?"+urllib.parse.urlencode({"action":"wbgetentities","ids":"|".join(ids),
        "props":"labels","languages":LANG,"format":"json"})
    req=urllib.request.Request(url, headers={"User-Agent":UA})
    for a in range(4):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                d=json.load(r)
            out={}
            for q,ent in d.get("entities",{}).items():
                lab=(ent.get("labels") or {}).get(LANG,{}).get("value")
                if lab: out[q]=lab
            return out
        except Exception:
            time.sleep(2*(a+1))
    print("  batch failed", file=sys.stderr); return {}
def main():
    L="/mnt/data3/jiaxuanluo/local_cache/term_memory/glossaries"
    for g in ("finance_wiki_12k","legal_wiki_12k"):
        d=json.load(open(f"{L}/{g}.json"))
        by_qid={}
        for e in d:
            q=e.get("wikidata_qid")
            if q: by_qid.setdefault(q,[]).append(e)
        qids=list(by_qid); lab_map={}
        for i in range(0,len(qids),50):
            lab_map.update(fetch(qids[i:i+50])); time.sleep(0.25)
            if (i//50)%20==0: print(f"  {g}: {i}/{len(qids)}, {len(lab_map)} {LANG}", flush=True)
        filled=0
        for q,es in by_qid.items():
            lab=lab_map.get(q)
            if lab:
                for e in es: e.setdefault("target_translations",{})[LANG]=lab; filled+=1
        json.dump(d, open(f"{L}/{g}.json","w"), ensure_ascii=False)
        print(f"{g}: filled {filled}/{len(d)} {LANG} ({100*filled//len(d)}%)", flush=True)
if __name__=="__main__": main()

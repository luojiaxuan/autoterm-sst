"""Fetch German, Japanese and Chinese Wikidata labels for the QIDs of topic slice glossaries.

Writes {qid: {lang: label}} for every QID found in the given glossary files.
"""

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://www.wikidata.org/w/api.php"
UA = "glossary-labels/0.1 (research script)"


def get(url):
    # Wikidata answers bursts with 429 and a Retry-After header.
    for attempt in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                return json.load(r)["entities"]
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 30) * (attempt + 1))
    raise RuntimeError("Wikidata kept answering 429")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glossaries", type=Path, nargs="+")
    ap.add_argument("--languages", default="de|ja|zh")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    qids = sorted({e["wikidata_qid"] for g in args.glossaries for e in json.loads(g.read_text()) if e.get("wikidata_qid")})
    labels = json.loads(args.out.read_text()) if args.out.exists() else {}
    todo = [q for q in qids if q not in labels]
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        query = urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(batch), "props": "labels",
                                        "languages": args.languages, "format": "json"})
        entities = get(f"{API}?{query}")
        for q in batch:
            labels[q] = {lang: v["value"] for lang, v in entities.get(q, {}).get("labels", {}).items()}
        args.out.write_text(json.dumps(labels, ensure_ascii=False))
        time.sleep(3.0)
    print(json.dumps({"qids": len(qids), "fetched": len(labels)}))


if __name__ == "__main__":
    main()

"""Add German and Japanese views of the 1k topic slices.

The MaxSim index encodes English source terms only, so another target language
reuses the en-zh embeddings. For each slice and language, an entry's target is
its stored translation in that language, else the Wikidata label of its QID
(fetch_wikidata_labels.py). Entries with neither are dropped, so the language's
index holds only rows that can produce a term pair. Each view is written as
<slice>/glossary.<lang>.json and <slice>/en-<lang>/maxsim.pt, and the manifest
gains an en-<lang> snapshot per slice.
"""

import argparse
import copy
import json
from pathlib import Path

import torch


def target(entry, lang, labels):
    stored = (entry.get("target_translations") or {}).get(lang)
    if stored:
        return stored
    return labels.get(entry.get("wikidata_qid") or "", {}).get(lang, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="slices folder holding manifest.json and prefix_1k/")
    ap.add_argument("--node-root", default="/data/inputs/slices", help="where the manifest says the folder lives")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--languages", nargs="+", default=["de", "ja"])
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    manifest = json.loads((args.root / "manifest.json").read_text(encoding="utf-8"))
    report = {}
    for slice_dir in sorted((args.root / "prefix_1k").iterdir()):
        if not (slice_dir / "glossary.json").exists():
            continue
        glossary = json.loads((slice_dir / "glossary.json").read_text(encoding="utf-8"))
        index = torch.load(slice_dir / "en-zh" / "maxsim.pt", map_location="cpu", weights_only=False)
        terms = [e["term"] if isinstance(e, dict) else str(e) for e in index["term_list"]]
        assert terms == [e["term"] for e in glossary], f"{slice_dir.name}: index rows do not follow the glossary"
        for lang in args.languages:
            rows, entries = [], []
            for i, entry in enumerate(glossary):
                t = target(entry, lang, labels)
                if t:
                    e = copy.deepcopy(entry)
                    e.setdefault("target_translations", {})[lang] = t
                    rows.append(i)
                    entries.append(e)
            view = {
                "text_embs": index["text_embs"][rows].clone(),
                "term_list": entries,
                "build_metadata": {**index.get("build_metadata", {}), "language_pair": f"en-{lang}",
                                   "term_count": len(rows), "derivation": f"en-zh rows with an {lang} target"},
            }
            (slice_dir / f"en-{lang}").mkdir(exist_ok=True)
            torch.save(view, slice_dir / f"en-{lang}" / "maxsim.pt")
            (slice_dir / f"glossary.{lang}.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
            base = f"{args.node_root}/prefix_1k/{slice_dir.name}"
            manifest["scales"][slice_dir.name][f"en-{lang}"] = {
                "terms_path": f"{base}/glossary.{lang}.json",
                "indexes": {"maxsim": f"{base}/en-{lang}/maxsim.pt"},
                "num_terms": len(rows),
            }
            report.setdefault(slice_dir.name, {})[lang] = len(rows)
    (args.root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

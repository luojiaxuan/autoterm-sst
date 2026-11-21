"""Topic slices that hold the RealSI talks' terminology, for the ten-domain RealSI stream.

Each RealSI en2zh talk belongs to one of the ten catalog domains. The annotated terms of
a talk (annotate_realsi_terms.py) become glossary entries of that domain's 1k slice, the
same way the NLP and medicine slices hold the ACL and ESO glossaries: a slice keeps 1,000
entries, the talk's terms first and then the slice's own entries in order, skipping any
whose source term a talk term already covers. A source term annotated with several
renderings keeps its most frequent one. Every slice index is encoded again with the
retriever's text encoder, and Merged-10k is the union of the ten slices without exact
duplicate (source, target) pairs, as for the original slices. The output folder mirrors
the slices folder and gets its own manifest.
"""

import argparse
import copy
import hashlib
import importlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

TALK_TO_PRESET = {
    "01-tech": "nlp_core_1k", "02-health": "medicine_core_1k", "03-edu": "education_core_1k",
    "04-fin": "finance_core_1k", "05-law": "legal_core_1k", "06-env": "environment_core_1k",
    "07-ent": "entertainment_core_1k", "08-sci": "science_core_1k", "09-sport": "sports_core_1k",
    "10-art": "art_core_1k",
}
FILLERS = {"um", "uh", "er", "erm", "ah", "hmm"}


def clean_source(text):
    """Spoken spans carry fillers and stutters ("health care care system"); glossary sources do not."""
    words = [w for w in re.split(r"\s+", text.strip()) if w.strip(",.").casefold() not in FILLERS]
    out = []
    for w in words:
        if not out or out[-1].casefold() != w.casefold():
            out.append(w)
    return " ".join(out).strip(" ,.")


def norm(text):
    return " ".join(text.casefold().split())


def talk_entries(talk):
    renderings = {}
    order = []
    for t in talk["terms"]:
        src = clean_source(t["src"])
        if not src:
            continue
        key = norm(src)
        if key not in renderings:
            renderings[key] = (src, Counter())
            order.append(key)
        renderings[key][1][t["trg"]] += 1
    entries = []
    for key in order:
        src, counts = renderings[key]
        trg = counts.most_common(1)[0][0]
        entries.append({"term": src, "term_key": key, "target_translations": {"zh": trg},
                        "source": "realsi_annotation", "realsi_talk": talk["talk"]})
    return entries


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=Path, required=True, help="folder with manifest.json and prefix_1k/")
    ap.add_argument("--terms", type=Path, required=True, help="output of annotate_realsi_terms.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--node-root", required=True, help="where the manifest says the output folder lives")
    ap.add_argument("--retriever-dir", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # The retriever imports helpers from the sibling common/ package of the RASST checkout.
    sys.path[:0] = [str(args.retriever_dir), str(args.retriever_dir.parent / "common")]
    builder = importlib.import_module("build_maxsim_index")
    device = torch.device(args.device)
    encoder, tokenizer = builder.build_text_encoder(device, builder.TEXT_LORA_RANK, builder.TEXT_LORA_ALPHA)
    builder.load_text_checkpoint(encoder, str(args.model_path), device)

    terms = json.load(open(args.terms, encoding="utf-8"))
    manifest = json.load(open(args.slices / "manifest.json", encoding="utf-8"))
    manifest["snapshot_id"] = "realsi_1k"
    manifest["source"] = "1k slices whose first entries are the RealSI talks' annotated terms"
    manifest["root"] = args.node_root
    report = {}
    slice_rows = {}
    for talk in terms["talks"]:
        preset = TALK_TO_PRESET[talk["domain"]]
        added = talk_entries(talk)
        covered = {e["term_key"] for e in added}
        base = json.load(open(args.slices / "prefix_1k" / preset / "glossary.json", encoding="utf-8"))
        kept = [e for e in base if norm(e["term"]) not in covered][: args.size - len(added)]
        glossary = added + kept
        folder = args.out / "prefix_1k" / preset
        (folder / "en-zh").mkdir(parents=True, exist_ok=True)
        gpath = folder / "glossary.json"
        gpath.write_text(json.dumps(glossary, ensure_ascii=False, indent=1), encoding="utf-8")
        term_list = builder.load_glossary(str(gpath))
        embs = builder.encode_terms(term_list, encoder, tokenizer, device).cpu()
        torch.save({"text_embs": embs, "term_list": term_list,
                    "build_metadata": {"language_pair": "en-zh", "term_count": len(term_list),
                                       "derivation": "RealSI talk terms then the base slice's entries"}},
                   folder / "en-zh" / "maxsim.pt")
        slice_rows[preset] = (glossary, embs)
        node = f"{args.node_root}/prefix_1k/{preset}"
        manifest["scales"][preset] = {"en-zh": {"terms_path": f"{node}/glossary.json",
                                                "indexes": {"maxsim": f"{node}/en-zh/maxsim.pt"},
                                                "num_terms": len(glossary)}}
        meta = manifest["preset_meta"][preset]
        meta.update({"glossary_path": f"{node}/glossary.json", "maxsim_index_path": f"{node}/en-zh/maxsim.pt",
                     "glossary_sha256": sha256(gpath), "term_count": len(glossary)})
        report[preset] = {"talk": talk["talk"], "realsi_terms": len(added),
                          "base_dropped_as_covered": sum(norm(e["term"]) in covered for e in base),
                          "size": len(glossary)}

    # Merged-10k: the ten slices in catalog order without exact duplicate (source, target) pairs.
    order = [p for p in manifest["scales"] if p.endswith("_core_1k")]
    seen, merged, rows = set(), [], []
    for preset in order:
        glossary, embs = slice_rows[preset]
        for i, e in enumerate(glossary):
            pair = (norm(e["term"]), (e.get("target_translations") or {}).get("zh", ""))
            if pair in seen:
                continue
            seen.add(pair)
            merged.append(e)
            rows.append(embs[i])
    folder = args.out / "prefix_1k" / "merged_10x1k"
    (folder / "en-zh").mkdir(parents=True, exist_ok=True)
    gpath = folder / "glossary.json"
    gpath.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    term_list = builder.load_glossary(str(gpath))
    torch.save({"text_embs": torch.stack(rows), "term_list": term_list,
                "build_metadata": {"language_pair": "en-zh", "term_count": len(merged),
                                   "derivation": "union of the ten RealSI slices"}}, folder / "en-zh" / "maxsim.pt")
    node = f"{args.node_root}/prefix_1k/merged_10x1k"
    manifest["scales"]["merged_10x1k"] = {"en-zh": {"terms_path": f"{node}/glossary.json",
                                                    "indexes": {"maxsim": f"{node}/en-zh/maxsim.pt"},
                                                    "num_terms": len(merged)}}
    manifest["preset_meta"]["merged_10x1k"].update({"glossary_path": f"{node}/glossary.json",
                                                    "maxsim_index_path": f"{node}/en-zh/maxsim.pt",
                                                    "glossary_sha256": sha256(gpath), "term_count": len(merged)})
    report["merged_10x1k"] = {"size": len(merged)}
    for preset in list(manifest["scales"]):
        if preset not in slice_rows and preset != "merged_10x1k":
            del manifest["scales"][preset]
            manifest["preset_meta"].pop(preset, None)
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

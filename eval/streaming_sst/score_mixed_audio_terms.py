#!/usr/bin/env python3
"""Score term accuracy for mixed ACL/medicine streaming JSON outputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TARGET_SAMPLE_RATE = 16000
DEFAULT_ACL_ROOT = "/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments"
DEFAULT_ACL_SOURCE_TEXT = "/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/acl_zh/source_text.txt"
DEFAULT_ACL_RAW_GLOSSARY = "/mnt/taurus/data2/jiaxuanluo/RASST/data/glossaries/acl6060_tagged_gt_raw_min_norm2.json"
DEFAULT_ACL_TECHNICAL_GOLD = str(PROJECT_ROOT / "eval/streaming_sst/acl_gold_technical.json")
DEFAULT_MEDICINE_ORACLE_DIR = "/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh"
_ALNUM_TERM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._+/#&%()-]*$")
_CJK_OR_KANA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def normalise_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def contains_cjk_or_kana(text: str) -> bool:
    return bool(_CJK_OR_KANA_RE.search(str(text or "")))


def allowed_identity_retention_source(term: str) -> bool:
    clean = normalise_space(term)
    if not clean:
        return False
    compact = clean.replace("-", "").replace("_", "")
    if len(compact) >= 2 and compact.upper() == compact and re.search(r"[A-Z]", compact):
        return True
    if " " not in clean and re.search(r"[A-Z].*[A-Z]", compact) and not clean.islower():
        return True
    if re.search(r"[A-Za-z]+\d|\d+[A-Za-z]", compact):
        return True
    return False


def output_contains_variant(output: str, variant: str) -> bool:
    variant = normalise_space(variant)
    if not variant:
        return False
    if _ALNUM_TERM_RE.fullmatch(variant):
        return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(variant) + r"(?![A-Za-z0-9])", output, re.IGNORECASE))
    return variant in output


def classify_output_hit(term: str, variants: Sequence[str], output: str) -> tuple[bool, str | None, str | None]:
    for variant in variants:
        if contains_cjk_or_kana(variant) and output_contains_variant(output, variant):
            return True, variant, "zh_translation"
    if allowed_identity_retention_source(term):
        for variant in variants:
            if not contains_cjk_or_kana(variant) and output_contains_variant(output, variant):
                kind = "acronym_retention" if term.upper() == term else "identity_retention"
                return True, variant, kind
    return False, None, None


@dataclass(frozen=True)
class GoldOccurrence:
    domain: str
    item_id: str
    block_index: int
    term: str
    variants: Sequence[str]
    source: str


def load_gold_entries(path: str, *, target_lang: str = "zh") -> List[tuple[str, List[str]]]:
    data = json.load(open(path, encoding="utf-8"))
    entries = data.values() if isinstance(data, dict) else data
    out: List[tuple[str, List[str]]] = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        term = normalise_space(item.get("en") or item.get("term") or "")
        variants: List[str] = []
        zh = item.get("zh") or item.get("variants") or item.get("acceptable_targets") or []
        if isinstance(zh, str):
            variants.append(normalise_space(zh))
        elif isinstance(zh, list):
            variants.extend(normalise_space(value) for value in zh)
        translations = item.get("target_translations")
        if isinstance(translations, dict):
            value = translations.get(target_lang)
            if isinstance(value, list):
                variants.extend(normalise_space(v) for v in value)
            elif value:
                variants.append(normalise_space(value))
        variants = [value for value in dict.fromkeys(variants) if value]
        if term and variants:
            out.append((term, variants))
    return out


def term_in_source(term: str, source_text: str) -> bool:
    term_norm = normalise_space(term)
    if not term_norm:
        return False
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._+/#&%()-]*", term_norm):
        return bool(re.search(r"(?<![A-Za-z0-9])" + re.escape(term_norm) + r"(?![A-Za-z0-9])", source_text, re.IGNORECASE))
    return term_norm.casefold() in source_text.casefold()


def load_acl_meta(acl_root: str) -> Dict[str, Dict[str, Any]]:
    meta_path = Path(acl_root) / "segments.meta.jsonl"
    by_wav: Dict[str, Dict[str, Any]] = {}
    for raw in meta_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        wav = str(row.get("seg_wav") or "")
        if wav:
            by_wav[wav] = row
    return by_wav


def selected_acl_source_text(block: Dict[str, Any], span: Dict[str, Any], *, acl_root: str, acl_source_text: str) -> str:
    by_wav = load_acl_meta(acl_root)
    source_lines = Path(acl_source_text).read_text(encoding="utf-8").splitlines()
    remaining = int(span["sample_count"])
    selected: List[str] = []
    for wav_path in block.get("wav_paths") or []:
        if remaining <= 0:
            break
        meta = by_wav.get(str(wav_path))
        if not meta:
            continue
        idx = int(meta.get("index") or meta.get("orig_index") or 0)
        if 0 <= idx < len(source_lines):
            selected.append(source_lines[idx])
        remaining -= int(round(float(meta.get("seg_duration") or meta.get("duration") or 0.0) * TARGET_SAMPLE_RATE))
    return "\n".join(selected)


def acl_occurrences(
    *,
    block: Dict[str, Any],
    span: Dict[str, Any],
    acl_root: str,
    acl_source_text: str,
    gold_entries: Sequence[tuple[str, Sequence[str]]],
    source_label: str,
) -> List[GoldOccurrence]:
    source_text = selected_acl_source_text(block, span, acl_root=acl_root, acl_source_text=acl_source_text)
    out: List[GoldOccurrence] = []
    for term, variants in gold_entries:
        if term_in_source(term, source_text):
            out.append(
                GoldOccurrence(
                    domain="nlp",
                    item_id=str(block["item_id"]),
                    block_index=int(span["block_index"]),
                    term=term,
                    variants=list(variants),
                    source=source_label,
                )
            )
    return out


def medicine_occurrences(block: Dict[str, Any], span: Dict[str, Any], *, oracle_dir: str) -> List[GoldOccurrence]:
    medicine_id = str(block["item_id"]).removeprefix("medicine_")
    oracle_path = Path(oracle_dir) / f"hard_medicine.oracle_term_map__medicine_{medicine_id}.json"
    rows = json.load(open(oracle_path, encoding="utf-8"))
    duration_s = float(span["sample_count"]) / TARGET_SAMPLE_RATE
    out: List[GoldOccurrence] = []
    for row in rows:
        if float(row.get("start_sec") or 0.0) >= duration_s:
            continue
        for ref in row.get("references") or []:
            term = normalise_space(ref.get("term") or "")
            translation = normalise_space(ref.get("translation") or "")
            if term and translation:
                out.append(
                    GoldOccurrence(
                        domain="medicine",
                        item_id=str(block["item_id"]),
                        block_index=int(span["block_index"]),
                        term=term,
                        variants=[translation],
                        source="medicine_oracle",
                    )
                )
    return out


def build_gold_sets(payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, List[GoldOccurrence]]:
    blocks = payload.get("blocks") or []
    spans = payload.get("block_spans") or []
    by_index = {int(span["block_index"]): span for span in spans}
    acl_raw = load_gold_entries(args.acl_raw_glossary, target_lang=args.target_lang)
    acl_technical = load_gold_entries(args.acl_technical_gold, target_lang=args.target_lang)
    gold_sets: Dict[str, List[GoldOccurrence]] = {"technical_plus_medicine": [], "raw_plus_medicine": []}
    for block_index, block in enumerate(blocks, start=1):
        span = by_index[block_index]
        if block.get("corpus") == "acl":
            gold_sets["technical_plus_medicine"].extend(
                acl_occurrences(
                    block=block,
                    span=span,
                    acl_root=args.acl_root,
                    acl_source_text=args.acl_source_text,
                    gold_entries=acl_technical,
                    source_label="acl_technical",
                )
            )
            gold_sets["raw_plus_medicine"].extend(
                acl_occurrences(
                    block=block,
                    span=span,
                    acl_root=args.acl_root,
                    acl_source_text=args.acl_source_text,
                    gold_entries=acl_raw,
                    source_label="acl_raw",
                )
            )
        elif block.get("corpus") == "medicine":
            med = medicine_occurrences(block, span, oracle_dir=args.medicine_oracle_dir)
            gold_sets["technical_plus_medicine"].extend(med)
            gold_sets["raw_plus_medicine"].extend(med)
    return gold_sets


def block_outputs(payload: Dict[str, Any]) -> Dict[int, str]:
    spans = payload.get("block_spans") or []
    records = payload.get("records") or []
    out: Dict[int, List[str]] = {int(span["block_index"]): [] for span in spans}
    for record in records:
        cursor = int(record.get("cursor_samples") or 0)
        for span in spans:
            if int(span["start_sample"]) < cursor <= int(span["end_sample"]):
                out[int(span["block_index"])].append(str(record.get("text") or record.get("text_preview") or ""))
                break
    return {idx: "".join(parts) for idx, parts in out.items()}


def score_occurrences(payload: Dict[str, Any], gold: Sequence[GoldOccurrence]) -> Dict[str, Any]:
    outputs = block_outputs(payload)
    traces: List[Dict[str, Any]] = []
    hit = 0
    by_domain: Dict[str, List[int]] = defaultdict(list)
    by_source: Dict[str, List[int]] = defaultdict(list)
    by_type: Dict[tuple[int, str, str, str, tuple[str, ...]], List[int]] = defaultdict(list)
    for occ in gold:
        output = outputs.get(occ.block_index, "")
        ok, variant, kind = classify_output_hit(occ.term, occ.variants, output)
        hit += int(ok)
        by_domain[occ.domain].append(int(ok))
        by_source[occ.source].append(int(ok))
        by_type[(occ.block_index, occ.domain, occ.source, occ.term, tuple(occ.variants))].append(int(ok))
        traces.append(
            {
                "domain": occ.domain,
                "item_id": occ.item_id,
                "block_index": occ.block_index,
                "term": occ.term,
                "variants": list(occ.variants),
                "source": occ.source,
                "hit": bool(ok),
                "hit_variant": variant,
                "hit_kind": kind,
            }
        )
    total = len(gold)
    type_metrics = summarize_type_metrics(by_type)
    return {
        "gold_occurrences": total,
        "hits": hit,
        "term_acc": round(hit / total, 4) if total else None,
        "unique_term_types": type_metrics["unique_term_types"],
        "type_hits_any": type_metrics["type_hits_any"],
        "type_acc_any": type_metrics["type_acc_any"],
        "type_hits_all": type_metrics["type_hits_all"],
        "type_acc_all": type_metrics["type_acc_all"],
        "by_domain": {
            domain: {
                "gold_occurrences": len(values),
                "hits": sum(values),
                "term_acc": round(sum(values) / len(values), 4) if values else None,
                **summarize_type_metrics(
                    {
                        key: type_hits
                        for key, type_hits in by_type.items()
                        if key[1] == domain
                    }
                ),
            }
            for domain, values in sorted(by_domain.items())
        },
        "by_source": {
            source: {
                "gold_occurrences": len(values),
                "hits": sum(values),
                "term_acc": round(sum(values) / len(values), 4) if values else None,
                **summarize_type_metrics(
                    {
                        key: type_hits
                        for key, type_hits in by_type.items()
                        if key[2] == source
                    }
                ),
            }
            for source, values in sorted(by_source.items())
        },
        "traces": traces,
    }


def summarize_type_metrics(grouped: Dict[tuple[int, str, str, str, tuple[str, ...]], Sequence[int]]) -> Dict[str, Any]:
    total = len(grouped)
    hit_any = sum(1 for values in grouped.values() if any(values))
    hit_all = sum(1 for values in grouped.values() if values and all(values))
    return {
        "unique_term_types": total,
        "type_hits_any": hit_any,
        "type_acc_any": round(hit_any / total, 4) if total else None,
        "type_hits_all": hit_all,
        "type_acc_all": round(hit_all / total, 4) if total else None,
    }


def parse_run(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"--run must be LABEL=PATH, got: {raw}")
    label, path = raw.split("=", 1)
    return label.strip(), path.strip()


def write_markdown(payload: Dict[str, Any], out_path: str) -> None:
    lines = ["# Mixed Audio Term Accuracy", ""]
    for gold_label, rows in payload["tables"].items():
        lines.extend(
            [
                f"## {gold_label}",
                "",
                "| run | term_acc | hits | gold | ACL acc | medicine acc | medicine type_acc_any | medicine type hits |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            by_domain = row["metrics"].get("by_domain") or {}
            acl = by_domain.get("nlp", {}).get("term_acc")
            medicine = by_domain.get("medicine", {}).get("term_acc")
            medicine_type_acc = by_domain.get("medicine", {}).get("type_acc_any")
            medicine_type_hits = by_domain.get("medicine", {}).get("type_hits_any")
            medicine_type_total = by_domain.get("medicine", {}).get("unique_term_types")
            medicine_type_cell = (
                f"{medicine_type_hits}/{medicine_type_total}"
                if medicine_type_hits is not None and medicine_type_total is not None
                else ""
            )
            metrics = row["metrics"]
            lines.append(
                f"| {row['run']} | {metrics.get('term_acc')} | {metrics.get('hits')} | "
                f"{metrics.get('gold_occurrences')} | {acl} | {medicine} | "
                f"{medicine_type_acc} | {medicine_type_cell} |"
            )
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="LABEL=path/to/mixed_eval.json")
    ap.add_argument("--acl-root", default=DEFAULT_ACL_ROOT)
    ap.add_argument("--acl-source-text", default=DEFAULT_ACL_SOURCE_TEXT)
    ap.add_argument("--acl-raw-glossary", default=DEFAULT_ACL_RAW_GLOSSARY)
    ap.add_argument("--acl-technical-gold", default=DEFAULT_ACL_TECHNICAL_GOLD)
    ap.add_argument("--medicine-oracle-dir", default=DEFAULT_MEDICINE_ORACLE_DIR)
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    runs = [(label, json.load(open(path, encoding="utf-8")), path) for label, path in map(parse_run, args.run)]
    gold_sets = build_gold_sets(runs[0][1], args)
    tables: Dict[str, List[Dict[str, Any]]] = {}
    for gold_label, gold in gold_sets.items():
        rows: List[Dict[str, Any]] = []
        for label, payload, path in runs:
            rows.append({"run": label, "path": path, "metrics": score_occurrences(payload, gold)})
        tables[gold_label] = rows

    output = {
        "gold_summary": {
            label: {
                "gold_occurrences": len(gold),
                "by_domain": dict((domain, sum(1 for occ in gold if occ.domain == domain)) for domain in sorted({occ.domain for occ in gold})),
                "by_source": dict((source, sum(1 for occ in gold if occ.source == source)) for source in sorted({occ.source for occ in gold})),
            }
            for label, gold in gold_sets.items()
        },
        "tables": tables,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(output, args.out_md)


if __name__ == "__main__":
    main()

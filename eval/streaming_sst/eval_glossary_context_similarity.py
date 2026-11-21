#!/usr/bin/env python3
"""Evaluate glossary-derived TF-IDF context similarity on RealSI text windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.streaming_sst.eval_realsi_domain_routing import DOMAIN_FILES, load_windows
from framework.agents.term_memory.domain_taxonomy import DOMAIN_TO_PRESET, WORKING_DOMAINS


def _translation(row: Dict[str, Any], target_lang: str) -> str:
    translations = row.get("target_translations")
    if isinstance(translations, dict):
        return str(translations.get(target_lang) or "").strip()
    return ""


def evaluate(
    *,
    catalog_dir: Path,
    realsi_root: Path,
    domains: List[str],
    target_lang: str,
    max_features: int,
    window_segments: int,
    step_segments: int,
) -> Dict[str, Any]:
    corpus: List[str] = []
    labels: List[str] = []
    for domain in domains:
        path = catalog_dir / f"{DOMAIN_TO_PRESET[domain]}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            text = _translation(row, target_lang)
            if text:
                corpus.append(text)
                labels.append(domain)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 4),
        min_df=2,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(corpus)
    labels_array = np.asarray(labels)
    centroids = normalize(
        np.vstack(
            [np.asarray(matrix[labels_array == domain].mean(axis=0)) for domain in domains]
        )
    )

    records: List[Dict[str, Any]] = []
    for domain in domains:
        if domain not in DOMAIN_FILES:
            continue
        windows = load_windows(
            realsi_root,
            domain,
            text_field="trg_text",
            window_segments=window_segments,
            step_segments=step_segments,
        )
        queries = vectorizer.transform([item.text for item in windows])
        similarities = np.asarray(queries @ centroids.T)
        for window, scores in zip(windows, similarities):
            order = np.argsort(-scores)
            predicted = domains[int(order[0])]
            records.append(
                {
                    "expected_domain": domain,
                    "predicted_domain": predicted,
                    "start_segment": window.start_segment,
                    "end_segment": window.end_segment,
                    "margin": round(float(scores[order[0]] - scores[order[1]]), 6),
                    "scores": {
                        candidate: round(float(scores[index]), 6)
                        for index, candidate in enumerate(domains)
                    },
                }
            )

    correct = sum(1 for item in records if item["expected_domain"] == item["predicted_domain"])
    per_domain = {}
    for domain in domains:
        selected = [item for item in records if item["expected_domain"] == domain]
        domain_correct = sum(1 for item in selected if item["predicted_domain"] == domain)
        per_domain[domain] = {
            "windows": len(selected),
            "correct": domain_correct,
            "accuracy": round(domain_correct / len(selected) if selected else 0.0, 4),
        }
    return {
        "domains": domains,
        "catalog_rows": len(corpus),
        "features": len(vectorizer.vocabulary_),
        "windows": len(records),
        "accuracy": round(correct / len(records) if records else 0.0, 4),
        "per_domain": per_domain,
        "records": records,
        "settings": {
            "target_lang": target_lang,
            "max_features": max_features,
            "window_segments": window_segments,
            "step_segments": step_segments,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog-dir", required=True, type=Path)
    ap.add_argument("--realsi-root", required=True, type=Path)
    ap.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--max-features", type=int, default=100_000)
    ap.add_argument("--window-segments", type=int, default=6)
    ap.add_argument("--step-segments", type=int, default=3)
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    ap.add_argument("--out-json", required=True, type=Path)
    ap.add_argument("--no-assert", action="store_true")
    args = ap.parse_args()

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    result = evaluate(
        catalog_dir=args.catalog_dir,
        realsi_root=args.realsi_root,
        domains=domains,
        target_lang=args.target_lang,
        max_features=args.max_features,
        window_segments=args.window_segments,
        step_segments=args.step_segments,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"windows={result['windows']} accuracy={result['accuracy']:.4f}")
    for domain, row in result["per_domain"].items():
        print(f"{domain:13s} {row['correct']}/{row['windows']} accuracy={row['accuracy']:.4f}")
    print(f"wrote {args.out_json}")
    if not args.no_assert and result["accuracy"] < args.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

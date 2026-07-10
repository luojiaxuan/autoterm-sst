#!/usr/bin/env python3
"""Build all AutoTerm MaxSim indexes while loading the text encoder once."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.domain_taxonomy import DOMAIN_TO_PRESET, WORKING_DOMAINS


def _load_builder(retriever_dir: Path):
    if not (retriever_dir / "build_maxsim_index.py").is_file():
        raise FileNotFoundError(f"build_maxsim_index.py not found under {retriever_dir}")
    sys.path.insert(0, str(retriever_dir))
    return importlib.import_module("build_maxsim_index")


def _work_items(
    catalog_dir: Path,
    output_root: Path,
    domains: List[str],
    *,
    target_lang: str,
    include_merged: bool,
) -> List[Tuple[str, Path, Path]]:
    items = [
        (
            DOMAIN_TO_PRESET[domain],
            catalog_dir / f"{DOMAIN_TO_PRESET[domain]}.json",
            output_root / DOMAIN_TO_PRESET[domain] / f"en-{target_lang}" / "maxsim.pt",
        )
        for domain in domains
    ]
    if include_merged:
        merged = sorted(catalog_dir.glob(f"merged_realsi_*_{target_lang}.json"))
        if len(merged) != 1:
            raise ValueError(f"expected one merged_realsi_*_{target_lang}.json, found {len(merged)}")
        merged_preset = merged[0].stem.removesuffix(f"_{target_lang}")
        items.append(
            (
                merged_preset,
                merged[0],
                output_root / merged_preset / f"en-{target_lang}" / "maxsim.pt",
            )
        )
    missing = [str(glossary) for _, glossary, _ in items if not glossary.is_file()]
    if missing:
        raise FileNotFoundError(f"missing glossary files: {missing}")
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever-dir", required=True, type=Path)
    ap.add_argument("--model-path", required=True, type=Path)
    ap.add_argument("--catalog-dir", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--domains", default=",".join(WORKING_DOMAINS))
    ap.add_argument("--target-lang", default="zh")
    ap.add_argument("--device", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--include-merged", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    unknown = [item for item in domains if item not in DOMAIN_TO_PRESET]
    if unknown:
        raise SystemExit(f"unknown domains: {', '.join(unknown)}")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    builder = _load_builder(args.retriever_dir)
    builder.TEXT_ENCODE_BATCH = args.batch_size
    device = torch.device(args.device)
    text_encoder, tokenizer = builder.build_text_encoder(
        device,
        builder.TEXT_LORA_RANK,
        builder.TEXT_LORA_ALPHA,
    )
    builder.load_text_checkpoint(text_encoder, str(args.model_path), device)
    items = _work_items(
        args.catalog_dir,
        args.output_root,
        domains,
        target_lang=args.target_lang,
        include_merged=args.include_merged,
    )

    results: List[Dict[str, Any]] = []
    for preset, glossary_path, output_path in items:
        if output_path.is_file() and not args.overwrite:
            print(f"[catalog-index] skip existing {preset}: {output_path}", flush=True)
            results.append(
                {
                    "preset": preset,
                    "glossary": str(glossary_path),
                    "index": str(output_path),
                    "status": "existing",
                    "bytes": output_path.stat().st_size,
                }
            )
            continue
        started = time.perf_counter()
        term_list = builder.load_glossary(str(glossary_path))
        text_embs = builder.encode_terms(term_list, text_encoder, tokenizer, device)
        if text_embs.shape[0] != len(term_list):
            raise RuntimeError(f"{preset}: embedding rows do not match term rows")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"text_embs": text_embs.cpu(), "term_list": term_list}, output_path)
        elapsed = time.perf_counter() - started
        result = {
            "preset": preset,
            "glossary": str(glossary_path),
            "index": str(output_path),
            "status": "built",
            "terms": len(term_list),
            "embedding_shape": list(text_embs.shape),
            "bytes": output_path.stat().st_size,
            "elapsed_s": round(elapsed, 3),
        }
        results.append(result)
        print(f"[catalog-index] {json.dumps(result, ensure_ascii=False)}", flush=True)
        del text_embs

    report = {
        "model_path": str(args.model_path),
        "device": args.device,
        "batch_size": args.batch_size,
        "catalog_dir": str(args.catalog_dir),
        "output_root": str(args.output_root),
        "results": results,
    }
    report_path = args.report or (args.output_root / "index_build_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[catalog-index] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()

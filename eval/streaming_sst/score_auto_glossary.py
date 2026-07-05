#!/usr/bin/env python3
"""Summarize adaptive working-glossary eval JSON as a paper-ready table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_rows(path: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"expected list JSON: {path}")
    return [row for row in data if isinstance(row, dict)]


def load_term_scores(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    rows = load_rows(path)
    return {str(row.get("preset")): row for row in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--auto-json", required=True, help="output from eval_auto_glossary.py")
    ap.add_argument("--term-json", default="", help="optional output from score_terms.py")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    term_scores = load_term_scores(args.term_json)
    rows = []
    for row in load_rows(args.auto_json):
        preset = str(row.get("preset") or "")
        term = term_scores.get(preset, {})
        rows.append(
            {
                "preset": preset,
                "term_recall": term.get("term_recall", ""),
                "false_copy": term.get("false_copy", ""),
                "bleu": term.get("bleu", ""),
                "masked_terms_bleu": term.get("masked_terms_bleu", ""),
                "refs/chunk": row.get("refs_per_chunk", ""),
                "prompt_refs/chunk": row.get("prompt_refs_per_chunk", ""),
                "candidate_pool/chunk": row.get("candidate_pool_per_chunk", ""),
                "shortfall_chunks": row.get("prompt_shortfall_chunks", ""),
                "rescue_chunks": row.get("open_wiki_rescue_chunks", ""),
                "fixed_prompt_k": row.get("fixed_prompt_k", ""),
                "prompt_gold@10": term.get("prompt_gold_retrieved_at_10", term.get("gold_retrieved", "")),
                "retr_precision@10": term.get("retrieval_precision_at_10", term.get("retrieval_precision", "")),
                "term_recall_surfaced": term.get("term_recall_surfaced", ""),
                "term_recall_not_surfaced": term.get("term_recall_not_surfaced", ""),
                "retrieve_p50_ms": row.get("retrieve_p50_ms", ""),
                "retrieve_p95_ms": row.get("retrieve_p95_ms", ""),
                "switches": row.get("switch_count", ""),
                "first_switch_s": row.get("first_switch_s", ""),
                "router_conf": row.get("router_confidence_avg", ""),
                "router_actions": ",".join(
                    item for item in dict.fromkeys(row.get("router_actions") or []) if item
                ),
            }
        )

    cols = [
        "preset",
        "term_recall",
        "false_copy",
        "bleu",
        "masked_terms_bleu",
        "refs/chunk",
        "prompt_refs/chunk",
        "candidate_pool/chunk",
        "shortfall_chunks",
        "rescue_chunks",
        "fixed_prompt_k",
        "prompt_gold@10",
        "retr_precision@10",
        "term_recall_surfaced",
        "term_recall_not_surfaced",
        "retrieve_p50_ms",
        "retrieve_p95_ms",
        "switches",
        "first_switch_s",
        "router_conf",
        "router_actions",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    table = "\n".join(lines)
    print(table)
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

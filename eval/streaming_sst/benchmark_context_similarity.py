#!/usr/bin/env python3
"""Benchmark AutoTerm context-similarity batches."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.agents.term_memory.context_similarity import DomainDescriptionSimilarity
from framework.agents.term_memory.domain_taxonomy import WORKING_DOMAINS


SAMPLE_TEXT = "患者接受临床治疗，医生根据诊断结果调整药物剂量并监测血压。"


async def run(args: argparse.Namespace) -> dict:
    scorer = DomainDescriptionSimilarity(
        model_id=args.model_id,
        device=args.device,
        batch_size=args.max_batch_size,
    )
    started = time.perf_counter()
    await scorer.start()
    load_s = time.perf_counter() - started
    rows = []
    for batch_size in args.batch_sizes:
        texts = [SAMPLE_TEXT] * batch_size
        await scorer.score_batch(texts, allowed_domains=WORKING_DOMAINS)
        samples = []
        for _ in range(args.repeats):
            tick = time.perf_counter()
            scores = await scorer.score_batch(texts, allowed_domains=WORKING_DOMAINS)
            samples.append(time.perf_counter() - tick)
            if len(scores) != batch_size:
                raise RuntimeError("context scorer returned the wrong batch size")
        ordered = sorted(samples)
        rows.append(
            {
                "batch_size": batch_size,
                "mean_ms": round(statistics.mean(samples) * 1000.0, 3),
                "p50_ms": round(statistics.median(samples) * 1000.0, 3),
                "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000.0, 3),
                "per_session_mean_ms": round(statistics.mean(samples) * 1000.0 / batch_size, 3),
            }
        )
    health = await scorer.health()
    await scorer.stop()
    return {
        "model_id": args.model_id,
        "device": args.device,
        "load_s": round(load_s, 3),
        "repeats": args.repeats,
        "health": health,
        "batches": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="BAAI/bge-m3")
    ap.add_argument("--device", required=True)
    ap.add_argument("--batch-sizes", default="1,8,32")
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()
    args.batch_sizes = [int(item) for item in args.batch_sizes.split(",") if item.strip()]
    result = asyncio.run(run(args))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

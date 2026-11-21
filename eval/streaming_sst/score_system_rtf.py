#!/usr/bin/env python3
"""Compute warm system RTF from mixed-stream evaluation JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SystemRtfResult:
    setting: str
    path: str
    audio_seconds: float
    wall_seconds: float
    system_rtf: float
    throughput_xrt: float
    routing_retrieval_seconds: float
    routing_retrieval_stage_ratio: float
    record_count: int
    final_cursor_samples: int


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--run requires nonempty LABEL and PATH")
    return label.strip(), Path(raw_path).expanduser()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON must be an object")
    return payload


def score_run(label: str, path: Path) -> SystemRtfResult:
    payload = load_payload(path)
    records = [row for row in payload.get("records", []) if isinstance(row, dict)]
    timed = [row for row in records if row.get("emitted_wall_s") is not None]
    if not timed:
        raise ValueError(f"{path}: no records contain emitted_wall_s")

    pacing = payload.get("pacing") or {}
    summary = payload.get("summary") or {}
    sent_samples = int(pacing.get("sent_samples") or 0)
    acknowledged_samples = int(pacing.get("acknowledged_cursor_samples") or 0)
    final_record = max(timed, key=lambda row: float(row["emitted_wall_s"]))
    final_cursor = int(final_record.get("cursor_samples") or 0)
    if sent_samples <= 0 or final_cursor != sent_samples or acknowledged_samples != sent_samples:
        raise ValueError(
            f"{path}: incomplete cursor coverage "
            f"(sent={sent_samples}, acknowledged={acknowledged_samples}, final={final_cursor})"
        )

    audio_seconds = float(summary.get("audio_seconds") or sent_samples / SAMPLE_RATE)
    sample_audio_seconds = sent_samples / SAMPLE_RATE
    if abs(audio_seconds - sample_audio_seconds) > 0.001:
        raise ValueError(
            f"{path}: summary audio {audio_seconds} disagrees with samples {sample_audio_seconds}"
        )
    wall_seconds = float(final_record["emitted_wall_s"])
    if wall_seconds <= 0:
        raise ValueError(f"{path}: final emitted wall time must be positive")

    retrieval_seconds = sum(
        float(row["retrieve_s"])
        for row in records
        if row.get("retrieve_s") is not None
    )
    return SystemRtfResult(
        setting=label,
        path=str(path),
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        system_rtf=wall_seconds / audio_seconds,
        throughput_xrt=audio_seconds / wall_seconds,
        routing_retrieval_seconds=retrieval_seconds,
        routing_retrieval_stage_ratio=retrieval_seconds / audio_seconds,
        record_count=len(records),
        final_cursor_samples=final_cursor,
    )


def write_tsv(results: Sequence[SystemRtfResult], path: Path | None) -> None:
    fieldnames = list(asdict(results[0]).keys())
    handle = path.open("w", encoding="utf-8", newline="") if path else None
    stream = handle if handle is not None else sys.stdout
    try:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            row = asdict(result)
            for key in (
                "audio_seconds",
                "wall_seconds",
                "system_rtf",
                "throughput_xrt",
                "routing_retrieval_seconds",
                "routing_retrieval_stage_ratio",
            ):
                row[key] = f"{float(row[key]):.6f}"
            writer.writerow(row)
    finally:
        if handle is not None:
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--out-tsv", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    results = [score_run(label, path) for label, path in args.run]
    if args.out_tsv:
        args.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(results, args.out_tsv)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps([asdict(result) for result in results], indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

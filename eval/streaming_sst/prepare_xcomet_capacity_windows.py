#!/usr/bin/env python3
"""Prepare paired ACL windows from glossary-capacity sweep outputs."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any, Sequence

from eval.streaming_sst.score_xcomet_windows import (
    ReferenceSegment,
    group_reference_segments,
    hypothesis_for_window,
    normalise_text,
    write_jsonl,
)


DEFAULT_TALKS = ",".join(
    [
        "2022.acl-long.268",
        "2022.acl-long.367",
        "2022.acl-long.590",
        "2022.acl-long.110",
        "2022.acl-long.117",
    ]
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--baseline-preset", default="acl_tagged_gs10k")
    parser.add_argument("--comparison-preset", required=True)
    parser.add_argument("--acl-meta", type=Path, required=True)
    parser.add_argument("--acl-source-text", type=Path, required=True)
    parser.add_argument("--acl-reference-text", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--talks", default=DEFAULT_TALKS)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audio_duration_s(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / handle.getframerate()


def prepare_windows(args: argparse.Namespace) -> list[dict[str, Any]]:
    run_rows = json.loads(args.runs_json.read_text(encoding="utf-8"))
    by_preset = {str(row["preset"]): row for row in run_rows}
    baseline = by_preset[args.baseline_preset]
    comparison = by_preset[args.comparison_preset]
    for row in (baseline, comparison):
        if not row.get("output_events"):
            raise ValueError(f"{row['preset']} has no output_events")
    if baseline.get("streaming_chunk_samples") != comparison.get("streaming_chunk_samples"):
        raise ValueError("capacity runs used different streaming chunk sizes")

    source_lines = args.acl_source_text.read_text(encoding="utf-8").splitlines()
    reference_lines = args.acl_reference_text.read_text(encoding="utf-8").splitlines()
    meta_rows = read_jsonl(args.acl_meta)
    by_talk: dict[str, list[dict[str, Any]]] = {}
    for row in meta_rows:
        by_talk.setdefault(str(row.get("talk") or ""), []).append(row)
    for rows in by_talk.values():
        rows.sort(key=lambda row: int(row.get("index") or row.get("orig_index") or 0))

    talks = [item.strip() for item in args.talks.split(",") if item.strip()]
    output: list[dict[str, Any]] = []
    block_start_sample = 0
    for block_index, talk in enumerate(talks, start=1):
        wav_path = args.audio_dir / f"{talk}.wav"
        block_duration_s = audio_duration_s(wav_path)
        segments: list[ReferenceSegment] = []
        for meta in by_talk.get(talk, []):
            index = int(meta.get("index") if meta.get("index") is not None else meta.get("orig_index", -1))
            if not 0 <= index < len(source_lines) or not 0 <= index < len(reference_lines):
                raise ValueError(f"ACL line index out of range: {index}")
            start_s = float(meta.get("offset") or 0.0)
            end_s = min(block_duration_s, start_s + float(meta.get("duration") or 0.0))
            if end_s <= start_s:
                continue
            segments.append(
                ReferenceSegment(
                    start_s=start_s,
                    end_s=end_s,
                    source=normalise_text(source_lines[index]),
                    reference=normalise_text(reference_lines[index]),
                )
            )
        grouped = group_reference_segments(
            segments,
            block_duration_s=block_duration_s,
            target_window_s=args.window_sec,
        )
        for window_index, window in enumerate(grouped, start=1):
            baseline_text, baseline_events = hypothesis_for_window(
                baseline["output_events"],
                block_start_sample=block_start_sample,
                local_start_s=float(window["local_start_s"]),
                local_end_s=float(window["local_end_s"]),
            )
            comparison_text, comparison_events = hypothesis_for_window(
                comparison["output_events"],
                block_start_sample=block_start_sample,
                local_start_s=float(window["local_start_s"]),
                local_end_s=float(window["local_end_s"]),
            )
            output.append(
                {
                    "block_index": block_index,
                    "window_index": window_index,
                    "item_id": talk,
                    "domain": "nlp",
                    "local_start_s": window["local_start_s"],
                    "local_end_s": window["local_end_s"],
                    "source": window["source"],
                    "reference": window["reference"],
                    "reference_segment_count": window["reference_segment_count"],
                    "auto_hypothesis": baseline_text,
                    "merged_hypothesis": comparison_text,
                    "auto_event_count": baseline_events,
                    "merged_event_count": comparison_events,
                    "baseline_preset": args.baseline_preset,
                    "comparison_preset": args.comparison_preset,
                }
            )
        block_start_sample += int(round(block_duration_s * 16000))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = prepare_windows(args)
    if not rows:
        raise SystemExit("no aligned windows prepared")
    write_jsonl(args.out_jsonl, rows)
    print(json.dumps({"windows": len(rows), "output": str(args.out_jsonl)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

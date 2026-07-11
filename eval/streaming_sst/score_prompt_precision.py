#!/usr/bin/env python3
"""Score time-local glossary prompt precision against MFA occurrences.

The mixed-audio streaming harness records the exact retrieved references shown
to the decoder for every output event, together with ``start_sample`` and
``cursor_samples``.  A reference is relevant when its normalized source term
matches a gold source-term occurrence whose MFA interval overlaps the
retrieval window for that event.  The default window is the current decoder
chunk plus the configured 1.92-second MaxSim lookback::

    [start_sample / 16000 - lookback_s, cursor_samples / 16000]

Every inserted reference is an independent precision decision.  A term that is
correctly retrieved in several chunks is therefore counted once per insertion;
gold occurrences are not consumed one-to-one as they are for output TERM_ACC.

By default the scorer requires ``len(record.references)`` to equal
``record.prompt_reference_count`` for every chunk.  This prevents a UI-truncated
reference list from being presented as decoder-prompt precision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.streaming_sst.score_time_aligned_terms import (  # noqa: E402
    TARGET_SAMPLE_RATE,
    TimedOccurrence,
    build_timed_gold,
    raw_annotation_count,
    term_tokens,
)


@dataclass(frozen=True)
class OccurrenceIndex:
    """MFA occurrence intervals keyed by normalized source term."""

    intervals_by_term: Mapping[str, Sequence[tuple[float, float]]]
    occurrence_count: int
    raw_annotation_rows: int

    @property
    def term_type_count(self) -> int:
        return len(self.intervals_by_term)


def normalise_source_term(value: Any) -> str:
    """Normalize glossary/MFA source terms to a shared token key."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(term_tokens(text))


def reference_source_term(reference: Mapping[str, Any]) -> str:
    """Return the first usable normalized source key from a prompt reference."""

    for field in ("key", "term_key", "term"):
        normalized = normalise_source_term(reference.get(field))
        if normalized:
            return normalized
    return ""


def build_occurrence_index(occurrences: Sequence[TimedOccurrence]) -> OccurrenceIndex:
    grouped: Dict[str, list[tuple[float, float]]] = {}
    for occurrence in occurrences:
        source_terms = (occurrence.term, *occurrence.source_aliases)
        keys = {normalise_source_term(term) for term in source_terms}
        interval = (float(occurrence.t_start), float(occurrence.t_end))
        for key in keys:
            if key:
                grouped.setdefault(key, []).append(interval)
    for key, intervals in grouped.items():
        grouped[key] = sorted(set(intervals))
    return OccurrenceIndex(
        grouped,
        len(occurrences),
        raw_annotation_count(occurrences),
    )


def interval_overlaps(
    occurrence: tuple[float, float],
    window: tuple[float, float],
    *,
    tolerance_s: float,
) -> bool:
    occurrence_start, occurrence_end = occurrence
    window_start, window_end = window
    tolerance = max(0.0, float(tolerance_s))
    return occurrence_start <= window_end + tolerance and occurrence_end >= window_start - tolerance


def reference_is_relevant(
    source_term: str,
    occurrence_index: OccurrenceIndex,
    window: tuple[float, float],
    *,
    tolerance_s: float,
) -> bool:
    if not source_term:
        return False
    return any(
        interval_overlaps(interval, window, tolerance_s=tolerance_s)
        for interval in occurrence_index.intervals_by_term.get(source_term, ())
    )


def playlist_signature(payload: Mapping[str, Any]) -> tuple[Any, ...]:
    blocks = payload.get("blocks") or []
    spans = payload.get("block_spans") or []
    if not blocks or not spans:
        raise ValueError("mixed-run payload must contain blocks and block_spans")
    block_signature = tuple(
        (
            str(block.get("item_id") or ""),
            str(block.get("corpus") or ""),
            str(block.get("expected_domain") or ""),
        )
        for block in blocks
    )
    span_signature = tuple(
        (
            int(span.get("block_index") or 0),
            str(span.get("item_id") or ""),
            int(span.get("start_sample") or 0),
            int(span.get("end_sample") or 0),
        )
        for span in spans
    )
    return block_signature, span_signature


def validate_same_playlist(payloads: Sequence[Mapping[str, Any]]) -> None:
    if not payloads:
        raise ValueError("at least one mixed-run payload is required")
    expected = playlist_signature(payloads[0])
    for index, payload in enumerate(payloads[1:], start=2):
        if playlist_signature(payload) != expected:
            raise ValueError(f"run {index} does not share the first run's playlist")


def timing_signature(payload: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(record.get("start_sample") or 0), int(record.get("cursor_samples") or 0))
        for record in (payload.get("records") or [])
        if isinstance(record, Mapping)
    )


def timing_signature_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(timing_signature(payload), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _precision(relevant: int, total: int) -> float | None:
    return round(relevant / total, 6) if total else None


def score_payload(
    payload: Mapping[str, Any],
    gold_sets: Mapping[str, Sequence[TimedOccurrence]],
    *,
    lookback_s: float,
    tolerance_s: float,
    require_complete_reference_capture: bool = True,
) -> Dict[str, Any]:
    """Score one mixed-run payload against one or more timed gold sets."""

    if lookback_s < 0.0:
        raise ValueError("lookback_s must be non-negative")
    if tolerance_s < 0.0:
        raise ValueError("tolerance_s must be non-negative")
    records = payload.get("records") or []
    if not isinstance(records, list) or not records:
        raise ValueError("mixed-run payload has no records")
    indexes = {label: build_occurrence_index(occurrences) for label, occurrences in gold_sets.items()}
    relevant = {label: 0 for label in indexes}
    total_prompt_references = 0
    total_captured_references = 0
    reference_count_mismatch_chunks = 0
    chunks_with_references = 0
    empty_source_references = 0

    for record_index, record in enumerate(records, start=1):
        if "start_sample" not in record or "cursor_samples" not in record:
            raise ValueError(f"record {record_index} lacks start_sample/cursor_samples")
        start_sample = int(record["start_sample"])
        cursor_sample = int(record["cursor_samples"])
        if start_sample < 0 or cursor_sample < start_sample:
            raise ValueError(
                f"record {record_index} has invalid sample window {start_sample}..{cursor_sample}"
            )
        if "prompt_reference_count" not in record:
            raise ValueError(f"record {record_index} lacks prompt_reference_count")
        prompt_count = int(record["prompt_reference_count"])
        if prompt_count < 0:
            raise ValueError(f"record {record_index} has a negative prompt_reference_count")
        raw_references = record.get("references") or []
        if not isinstance(raw_references, list) or any(not isinstance(item, dict) for item in raw_references):
            raise ValueError(f"record {record_index}.references must be a list of dictionaries")
        references = [dict(item) for item in raw_references]
        if len(references) != prompt_count:
            reference_count_mismatch_chunks += 1
            if require_complete_reference_capture:
                raise ValueError(
                    f"record {record_index} captured {len(references)} references but "
                    f"prompt_reference_count={prompt_count}"
                )

        total_prompt_references += prompt_count
        total_captured_references += len(references)
        chunks_with_references += int(bool(references))
        window = (
            max(0.0, start_sample / TARGET_SAMPLE_RATE - float(lookback_s)),
            cursor_sample / TARGET_SAMPLE_RATE,
        )
        for reference in references:
            source_term = reference_source_term(reference)
            if not source_term:
                empty_source_references += 1
            for label, occurrence_index in indexes.items():
                relevant[label] += int(
                    reference_is_relevant(
                        source_term,
                        occurrence_index,
                        window,
                        tolerance_s=tolerance_s,
                    )
                )

    chunk_count = len(records)
    metrics: Dict[str, Any] = {
        "preset": str(
            (payload.get("config") or {}).get("preset")
            or (payload.get("summary") or {}).get("preset")
            or ""
        ),
        "chunk_count": chunk_count,
        "emitted_translation_chunks": chunk_count,
        "chunks_with_references": chunks_with_references,
        "prompt_reference_count": total_prompt_references,
        "captured_reference_count": total_captured_references,
        "reference_count_mismatch_chunks": reference_count_mismatch_chunks,
        "empty_source_reference_count": empty_source_references,
        "retrieved_references_per_chunk": round(total_prompt_references / chunk_count, 6),
        "captured_references_per_chunk": round(total_captured_references / chunk_count, 6),
        "reference_capture_ratio": (
            round(total_captured_references / total_prompt_references, 6)
            if total_prompt_references
            else 1.0
        ),
    }
    for label, occurrence_index in indexes.items():
        metrics[label] = {
            "gold_occurrences": occurrence_index.occurrence_count,
            "raw_annotation_rows": occurrence_index.raw_annotation_rows,
            "gold_source_term_types": occurrence_index.term_type_count,
            "relevant_references": relevant[label],
            "evaluated_references": total_captured_references,
            "prompt_precision": _precision(relevant[label], total_captured_references),
        }
    return metrics


def parse_run(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--run must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"--run must be NAME=PATH, got {spec!r}")
    return name.strip(), path.strip()


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    run_specs = [parse_run(spec) for spec in args.run]
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for _, path in run_specs]
    validate_same_playlist(payloads)
    gold_sets = build_timed_gold(payloads[0], args)
    timing_signatures = [timing_signature(payload) for payload in payloads]
    report: Dict[str, Any] = {
        "protocol": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "lookback_s": float(args.lookback_s),
            "alignment_tolerance_s": float(args.alignment_tolerance_s),
            "reference_capture_required": not bool(args.allow_reference_count_mismatch),
            "relevance": "normalized reference source term matches an overlapping MFA source occurrence",
            "refs_per_chunk_denominator": "emitted translation events; empty-output decoder ticks are not persisted",
            "timing_signatures_identical": all(
                signature == timing_signatures[0] for signature in timing_signatures[1:]
            ),
        },
        "runs": {},
    }
    for (name, path), payload in zip(run_specs, payloads):
        row = score_payload(
            payload,
            gold_sets,
            lookback_s=args.lookback_s,
            tolerance_s=args.alignment_tolerance_s,
            require_complete_reference_capture=not args.allow_reference_count_mismatch,
        )
        row["path"] = path
        row["timing_signature_sha256"] = timing_signature_sha256(payload)
        report["runs"][name] = row
    return report


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Time-local Prompt Precision",
        "",
        f"- MaxSim lookback: `{report['protocol']['lookback_s']}s`",
        f"- MFA overlap tolerance: `{report['protocol']['alignment_tolerance_s']}s`",
        f"- Identical event timing across runs: `{report['protocol']['timing_signatures_identical']}`",
        "",
        "| run | preset | technical precision | raw precision | refs/chunk | captured/prompt | mismatch chunks |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["runs"].items():
        technical = row.get("technical_plus_medicine") or {}
        raw = row.get("raw_plus_medicine") or {}
        lines.append(
            f"| {name} | {row.get('preset', '')} | {technical.get('prompt_precision')} | "
            f"{raw.get('prompt_precision')} | {row.get('retrieved_references_per_chunk')} | "
            f"{row.get('reference_capture_ratio')} | {row.get('reference_count_mismatch_chunks')} |"
        )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--mfa-root", required=True)
    parser.add_argument("--acl-root", required=True)
    parser.add_argument("--acl-technical-gold", required=True)
    parser.add_argument("--acl-raw-glossary", required=True)
    parser.add_argument("--medicine-oracle-dir", required=True)
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument("--lookback-s", type=float, default=1.92)
    parser.add_argument("--alignment-tolerance-s", type=float, default=0.0)
    parser.add_argument("--allow-reference-count-mismatch", action="store_true")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown_report(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score prompt-reference volume and precision for glossary-capacity runs.

The mixed-audio runner persists the UI reference list together with
``prompt_reference_count``.  The prompt list is the first N entries of that
same ranked list, so this scorer reconstructs only those prompt-injected
references.  It reports observation coverage when an older/incomplete JSON
does not contain all declared prompt references.

Two precision views are intentionally named separately:

* gold-type prompt precision: a reference event is correct when its source
  term belongs to the curated ACL technical/raw inventory;
* source-time-aligned prompt precision: in addition, a matching curated term
  must occur in the source-audio window used for that retrieval event.

The second view reuses ``score_time_aligned_terms.build_timed_gold`` for all
TextGrid-to-playlist alignment.  It is emitted as unavailable, rather than
silently falling back to type precision, when the required alignment assets
are missing.  Missing per-record timing is reflected in alignment coverage.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.streaming_sst.score_mixed_audio_terms import (
    DEFAULT_ACL_RAW_GLOSSARY,
    DEFAULT_ACL_ROOT,
    DEFAULT_ACL_TECHNICAL_GOLD,
    DEFAULT_MEDICINE_ORACLE_DIR,
    load_gold_entries,
    normalise_space,
)
from eval.streaming_sst.score_time_aligned_terms import (
    TARGET_SAMPLE_RATE,
    TimedOccurrence,
    build_timed_gold,
)

DEFAULT_MFA_ROOT = str(PROJECT_ROOT / "eval/streaming_sst/mfa_alignments")
DEFAULT_RETRIEVAL_LOOKBACK_S = 1.92


@dataclass(frozen=True)
class PromptReferenceEvent:
    """One persisted reference that was actually inserted into a prompt."""

    record_index: int
    record: Mapping[str, Any]
    reference: Mapping[str, Any]


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def normalise_source_term(value: Any) -> str:
    """Canonicalise source-side glossary labels for exact type matching."""
    return unicodedata.normalize("NFKC", normalise_space(str(value or ""))).casefold()


def _reference_term(reference: Mapping[str, Any]) -> str:
    return normalise_source_term(reference.get("term") or reference.get("source_label") or "")


def _nonnegative_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def collect_prompt_reference_events(
    payload: Mapping[str, Any],
) -> tuple[List[PromptReferenceEvent], Dict[str, Any]]:
    """Reconstruct prompt references and quantify persistence completeness.

    ``meta.references`` can be longer than the actual prompt list when the UI
    top-k exceeds the prompt top-k.  Conversely, older run JSONs may omit some
    or all references.  Only the first ``prompt_reference_count`` valid dicts
    are scored when that declaration is available.
    """
    records = [row for row in (payload.get("records") or []) if isinstance(row, Mapping)]
    events: List[PromptReferenceEvent] = []
    declared_chunks = 0
    declared_events = 0
    observed_declared_events = 0
    complete_declared_chunks = 0
    inferred_events = 0
    malformed_saved_entries = 0

    for record_index, record in enumerate(records):
        raw_refs = record.get("references") or []
        if not isinstance(raw_refs, list):
            raw_refs = []
        saved_refs = [ref for ref in raw_refs if isinstance(ref, Mapping)]
        malformed_saved_entries += len(raw_refs) - len(saved_refs)
        declared = _nonnegative_int(record.get("prompt_reference_count"))
        if declared is None:
            selected = saved_refs
            inferred_events += len(selected)
        else:
            declared_chunks += 1
            declared_events += declared
            observed_count = min(declared, len(saved_refs))
            observed_declared_events += observed_count
            complete_declared_chunks += int(len(saved_refs) >= declared)
            selected = saved_refs[:declared]
        events.extend(
            PromptReferenceEvent(record_index=record_index, record=record, reference=reference)
            for reference in selected
        )

    chunks = len(records)
    observed_events = len(events)
    volume = {
        "chunks": chunks,
        "observed_prompt_reference_events": observed_events,
        "retrieved_references_per_chunk": (
            round(observed_events / chunks, 6) if chunks else None
        ),
        "chunks_with_declared_prompt_reference_count": declared_chunks,
        "declared_prompt_reference_events": declared_events,
        "observed_declared_prompt_reference_events": observed_declared_events,
        "missing_declared_prompt_reference_events": max(
            0, declared_events - observed_declared_events
        ),
        "prompt_reference_observation_coverage": _ratio(
            observed_declared_events, declared_events
        ),
        "complete_declared_prompt_reference_chunks": complete_declared_chunks,
        "complete_declared_prompt_reference_chunk_rate": _ratio(
            complete_declared_chunks, declared_chunks
        ),
        "prompt_reference_events_inferred_without_count": inferred_events,
        "malformed_saved_reference_entries": malformed_saved_entries,
    }
    return events, volume


def load_source_term_inventory(path: str, *, target_lang: str) -> set[str]:
    terms = (
        normalise_source_term(term)
        for term, _ in load_gold_entries(path, target_lang=target_lang)
    )
    return {term for term in terms if term}


def score_gold_type_prompt_precision(
    events: Sequence[PromptReferenceEvent], gold_terms: set[str]
) -> Dict[str, Any]:
    denominator = len(events)
    refs_with_term = sum(bool(_reference_term(event.reference)) for event in events)
    hits = sum(_reference_term(event.reference) in gold_terms for event in events)
    return {
        "inventory_term_types": len(gold_terms),
        "prompt_reference_events": denominator,
        "prompt_reference_events_with_source_term": refs_with_term,
        "source_term_field_coverage": _ratio(refs_with_term, denominator),
        "gold_type_matches": hits,
        "gold_type_prompt_precision": _ratio(hits, denominator),
    }


def _span_and_block_maps(
    payload: Mapping[str, Any],
) -> tuple[Dict[int, Mapping[str, Any]], Dict[int, Mapping[str, Any]]]:
    spans = {
        int(span["block_index"]): span
        for span in (payload.get("block_spans") or [])
        if isinstance(span, Mapping) and "block_index" in span
    }
    blocks = {
        block_index: block
        for block_index, block in enumerate(payload.get("blocks") or [], start=1)
        if isinstance(block, Mapping)
    }
    return spans, blocks


def _record_block_index(
    record: Mapping[str, Any], spans: Mapping[int, Mapping[str, Any]]
) -> int | None:
    if "cursor_samples" not in record:
        return None
    cursor = _nonnegative_int(record.get("cursor_samples"))
    if cursor is None:
        return None
    for block_index, span in spans.items():
        try:
            if int(span["start_sample"]) < cursor <= int(span["end_sample"]):
                return block_index
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _timed_occurrence_index(
    occurrences: Iterable[TimedOccurrence],
) -> Dict[tuple[int, str], List[TimedOccurrence]]:
    out: Dict[tuple[int, str], List[TimedOccurrence]] = {}
    for occurrence in occurrences:
        # note (luojiaxuan): Capacity sweeps are ACL-only; exclude medicine
        # oracle occurrences so the ACL label stays honest on mixed files.
        if occurrence.domain != "nlp":
            continue
        key = (int(occurrence.block_index), normalise_source_term(occurrence.term))
        if key[1]:
            out.setdefault(key, []).append(occurrence)
    return out


def _reference_query_window(
    event: PromptReferenceEvent,
    *,
    span: Mapping[str, Any],
    retrieval_lookback_s: float,
) -> tuple[float, float] | None:
    record = event.record
    if "start_sample" not in record or "cursor_samples" not in record:
        return None
    start = _nonnegative_int(record.get("start_sample"))
    cursor = _nonnegative_int(record.get("cursor_samples"))
    if start is None or cursor is None or cursor < start:
        return None
    try:
        block_start = int(span["start_sample"])
        block_end = int(span["end_sample"])
    except (KeyError, TypeError, ValueError):
        return None
    lookback_samples = max(0, int(round(retrieval_lookback_s * TARGET_SAMPLE_RATE)))
    query_start = max(block_start, start - lookback_samples)
    query_end = min(block_end, cursor)
    if query_end < query_start:
        return None
    return query_start / TARGET_SAMPLE_RATE, query_end / TARGET_SAMPLE_RATE


def score_source_time_aligned_prompt_precision(
    payload: Mapping[str, Any],
    events: Sequence[PromptReferenceEvent],
    occurrences: Sequence[TimedOccurrence],
    *,
    retrieval_lookback_s: float,
) -> Dict[str, Any]:
    spans, blocks = _span_and_block_maps(payload)
    occurrence_index = _timed_occurrence_index(occurrences)
    eligible = 0
    hits = 0
    excluded_non_acl = 0
    excluded_missing_timing_or_span = 0

    for event in events:
        block_index = _record_block_index(event.record, spans)
        if block_index is None or block_index not in blocks:
            excluded_missing_timing_or_span += 1
            continue
        if str(blocks[block_index].get("corpus") or "") != "acl":
            excluded_non_acl += 1
            continue
        window = _reference_query_window(
            event,
            span=spans[block_index],
            retrieval_lookback_s=retrieval_lookback_s,
        )
        if window is None:
            excluded_missing_timing_or_span += 1
            continue
        eligible += 1
        term = _reference_term(event.reference)
        query_start_s, query_end_s = window
        hits += int(
            any(
                occurrence.t_end >= query_start_s and occurrence.t_start <= query_end_s
                for occurrence in occurrence_index.get((block_index, term), [])
            )
        )

    observed = len(events)
    occurrence_types = {key for key in occurrence_index}
    return {
        "gold_occurrences": sum(len(rows) for rows in occurrence_index.values()),
        "gold_occurrence_term_types_by_block": len(occurrence_types),
        "observed_prompt_reference_events": observed,
        "eligible_prompt_reference_events": eligible,
        "excluded_non_acl_prompt_reference_events": excluded_non_acl,
        "excluded_missing_timing_or_span_prompt_reference_events": (
            excluded_missing_timing_or_span
        ),
        "source_time_alignment_coverage": _ratio(eligible, observed),
        "source_time_aligned_matches": hits,
        "source_time_aligned_prompt_precision": _ratio(hits, eligible),
    }


def _alignment_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mfa_root=args.mfa_root,
        acl_root=args.acl_root,
        acl_technical_gold=args.acl_technical_gold,
        acl_raw_glossary=args.acl_raw_glossary,
        medicine_oracle_dir=args.medicine_oracle_dir,
        target_lang=args.target_lang,
    )


def evaluate_run(payload: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    events, volume = collect_prompt_reference_events(payload)
    technical_terms = load_source_term_inventory(
        args.acl_technical_gold, target_lang=args.target_lang
    )
    raw_terms = load_source_term_inventory(args.acl_raw_glossary, target_lang=args.target_lang)
    result: Dict[str, Any] = {
        "reference_volume": volume,
        "gold_type_prompt_precision": {
            "acl_technical_gold": score_gold_type_prompt_precision(events, technical_terms),
            "acl_raw_gold": score_gold_type_prompt_precision(events, raw_terms),
        },
    }

    if getattr(args, "no_source_time_alignment", False):
        result["source_time_aligned_prompt_precision"] = {
            "available": False,
            "reason": "disabled by --no-source-time-alignment",
        }
        return result

    try:
        timed_gold = build_timed_gold(dict(payload), _alignment_args(args))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["source_time_aligned_prompt_precision"] = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "observed_prompt_reference_events": len(events),
        }
        return result

    result["source_time_aligned_prompt_precision"] = {
        "available": True,
        "retrieval_lookback_s": args.retrieval_lookback_s,
        "acl_technical_gold": score_source_time_aligned_prompt_precision(
            payload,
            events,
            timed_gold["technical_plus_medicine"],
            retrieval_lookback_s=args.retrieval_lookback_s,
        ),
        "acl_raw_gold": score_source_time_aligned_prompt_precision(
            payload,
            events,
            timed_gold["raw_plus_medicine"],
            retrieval_lookback_s=args.retrieval_lookback_s,
        ),
    }
    return result


def parse_run_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"--run must be NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError(f"--run must be NAME=PATH, got {spec!r}")
    return name, path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--acl-technical-gold", default=DEFAULT_ACL_TECHNICAL_GOLD)
    parser.add_argument("--acl-raw-glossary", default=DEFAULT_ACL_RAW_GLOSSARY)
    parser.add_argument("--mfa-root", default=DEFAULT_MFA_ROOT)
    parser.add_argument("--acl-root", default=DEFAULT_ACL_ROOT)
    parser.add_argument("--medicine-oracle-dir", default=DEFAULT_MEDICINE_ORACLE_DIR)
    parser.add_argument("--target-lang", default="zh")
    parser.add_argument(
        "--retrieval-lookback-s", type=float, default=DEFAULT_RETRIEVAL_LOOKBACK_S
    )
    parser.add_argument("--no-source-time-alignment", action="store_true")
    parser.add_argument("--out-json", default="")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.retrieval_lookback_s < 0:
        raise SystemExit("--retrieval-lookback-s must be non-negative")
    output: Dict[str, Any] = {
        "schema_version": 1,
        "metric_definitions": {
            "retrieved_references_per_chunk": (
                "Persisted prompt-injected reference events divided by output chunks. "
                "When prompt_reference_count is present, only the first N ranked "
                "references are included."
            ),
            "gold_type_prompt_precision": (
                "Prompt reference events whose normalized source term is in the named "
                "curated ACL inventory, divided by observed prompt reference events."
            ),
            "source_time_aligned_prompt_precision": (
                "Eligible prompt reference events whose curated source term has an MFA "
                "occurrence overlapping [chunk_start-lookback, chunk_end], divided by "
                "eligible prompt reference events."
            ),
        },
        "runs": {},
    }
    for spec in args.run:
        name, path = parse_run_spec(spec)
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        output["runs"][name] = {"input_path": path, **evaluate_run(payload, args)}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

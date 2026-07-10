#!/usr/bin/env python3
"""Compose fixed-domain mixed-audio runs into one blockwise-oracle payload.

The common use case is a fixed-NLP run containing two ACL talks and a separate
fixed-medicine run.  Blocks can be selected and reordered without rerunning the
model::

    python eval/streaming_sst/stitch_blockwise_oracle_runs.py \\
      --block acl_fixed.json=1 \\
      --block medicine_fixed.json=1 \\
      --block acl_fixed.json=2 \\
      --out-json blockwise_oracle.json

Each selected block is clipped from its source run, rebased to zero, and then
packed onto a new contiguous sample timeline.  Full payloads can instead be
given as positional arguments; all of their blocks are expanded in order.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TARGET_SAMPLE_RATE = 16000

_RUN_SPECIFIC_CONFIG_KEYS = {
    "base_url",
    "medicine_ids",
    "preset",
    "schedule",
    "seed",
}
_STRICT_SHARED_CONFIG_KEYS = (
    "language_pair",
    "latency_multiplier",
    "feed_sleep",
    "max_seconds_per_item",
)
_REQUIRED_SHARED_CONFIG_KEYS = ("language_pair", "latency_multiplier")
_POSITION_TOKENS = {"start", "end", "cursor", "boundary", "position", "offset", "first", "last"}


@dataclass(frozen=True)
class BlockSelection:
    """One source block to append to the composed payload."""

    payload: Mapping[str, Any]
    block_index: int
    source_label: str
    source_sha256: str = ""


@dataclass(frozen=True)
class ValidatedSource:
    payload: Mapping[str, Any]
    label: str
    sha256: str
    config: Mapping[str, Any]
    blocks: Tuple[Mapping[str, Any], ...]
    spans: Tuple[Mapping[str, Any], ...]
    records_by_block: Mapping[int, Tuple[Mapping[str, Any], ...]]
    chunk_samples: int


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer, not a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{context} must be an integer")
    return result


def _required_text(row: Mapping[str, Any], key: str, context: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{context}.{key} must be non-empty")
    return value


def _block_identity(row: Mapping[str, Any], context: str) -> Tuple[str, str, str]:
    return (
        _required_text(row, "item_id", context),
        _required_text(row, "corpus", context),
        _required_text(row, "expected_domain", context),
    )


def _record_block_index(cursor_samples: int, spans: Sequence[Mapping[str, Any]]) -> Optional[int]:
    for span in spans:
        if int(span["start_sample"]) < cursor_samples <= int(span["end_sample"]):
            return int(span["block_index"])
    return None


def _resolve_chunk_samples(payload: Mapping[str, Any], label: str) -> int:
    config = payload.get("config") or {}
    summary = payload.get("summary") or {}
    if not isinstance(config, Mapping) or not isinstance(summary, Mapping):
        raise ValueError(f"{label}: config and summary must be dictionaries")
    raw = config.get("chunk_samples", summary.get("chunk_samples"))
    chunk_samples = _integer(raw, f"{label}.config.chunk_samples")
    if chunk_samples <= 0:
        raise ValueError(f"{label}.config.chunk_samples must be positive")
    return chunk_samples


def validate_source_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    sha256: str = "",
) -> ValidatedSource:
    """Validate one original eval_mixed_audio_switch payload."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}: top-level JSON value must be a dictionary")
    raw_blocks = payload.get("blocks")
    raw_spans = payload.get("block_spans")
    raw_records = payload.get("records")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError(f"{label}: blocks must be a non-empty list")
    if not isinstance(raw_spans, list) or len(raw_spans) != len(raw_blocks):
        raise ValueError(f"{label}: block_spans must contain exactly one span per block")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"{label}: records must be a non-empty list")
    if any(not isinstance(item, Mapping) for item in raw_blocks):
        raise ValueError(f"{label}: every block must be a dictionary")
    if any(not isinstance(item, Mapping) for item in raw_spans):
        raise ValueError(f"{label}: every block span must be a dictionary")
    if any(not isinstance(item, Mapping) for item in raw_records):
        raise ValueError(f"{label}: every record must be a dictionary")

    span_by_index: Dict[int, Mapping[str, Any]] = {}
    for position, span in enumerate(raw_spans, start=1):
        index = _integer(span.get("block_index"), f"{label}.block_spans[{position}].block_index")
        if index in span_by_index:
            raise ValueError(f"{label}: duplicate block_index {index}")
        span_by_index[index] = span
    expected_indices = set(range(1, len(raw_blocks) + 1))
    if set(span_by_index) != expected_indices:
        raise ValueError(f"{label}: block indices must be exactly 1..{len(raw_blocks)}")
    spans = tuple(span_by_index[index] for index in range(1, len(raw_blocks) + 1))

    previous_end = 0
    identities: set[Tuple[str, str]] = set()
    for index, (block, span) in enumerate(zip(raw_blocks, spans), start=1):
        block_identity = _block_identity(block, f"{label}.blocks[{index}]")
        span_identity = _block_identity(span, f"{label}.block_spans[{index}]")
        if block_identity != span_identity:
            raise ValueError(
                f"{label}: block {index} identity {block_identity!r} does not match span identity {span_identity!r}"
            )
        short_identity = (block_identity[1], block_identity[0])
        if short_identity in identities:
            raise ValueError(f"{label}: duplicate source block identity {short_identity!r}")
        identities.add(short_identity)
        start = _integer(span.get("start_sample"), f"{label}.block_spans[{index}].start_sample")
        end = _integer(span.get("end_sample"), f"{label}.block_spans[{index}].end_sample")
        if start < 0 or end <= start:
            raise ValueError(f"{label}: block {index} has invalid span {start}..{end}")
        if index == 1 and start != 0:
            raise ValueError(f"{label}: first block must start at sample 0, got {start}")
        if index > 1 and start < previous_end:
            raise ValueError(f"{label}: block {index} overlaps the preceding block")
        previous_end = end
        if "sample_count" in span:
            sample_count = _integer(span["sample_count"], f"{label}.block_spans[{index}].sample_count")
            if sample_count != end - start:
                raise ValueError(
                    f"{label}: block {index} sample_count {sample_count} does not equal span length {end - start}"
                )
        if "wav_count" in span and isinstance(block.get("wav_paths"), (list, tuple)):
            wav_count = _integer(span["wav_count"], f"{label}.block_spans[{index}].wav_count")
            if wav_count != len(block["wav_paths"]):
                raise ValueError(f"{label}: block {index} wav_count does not match block.wav_paths")

    records_by_block: Dict[int, List[Mapping[str, Any]]] = {index: [] for index in expected_indices}
    previous_cursor = -1
    for position, record in enumerate(raw_records, start=1):
        if "start_sample" not in record or "cursor_samples" not in record:
            raise ValueError(f"{label}: record {position} lacks start_sample/cursor_samples")
        start = _integer(record["start_sample"], f"{label}.records[{position}].start_sample")
        cursor = _integer(record["cursor_samples"], f"{label}.records[{position}].cursor_samples")
        if start < 0 or cursor < start:
            raise ValueError(f"{label}: record {position} has invalid sample window {start}..{cursor}")
        if cursor < previous_cursor:
            raise ValueError(f"{label}: record cursor_samples are not monotonic")
        previous_cursor = cursor
        block_index = _record_block_index(cursor, spans)
        if block_index is None:
            raise ValueError(f"{label}: record {position} cursor {cursor} falls outside all block spans")
        expected_domain = str(spans[block_index - 1]["expected_domain"])
        recorded_expected = str(record.get("expected_domain") or "")
        if recorded_expected and recorded_expected != expected_domain:
            raise ValueError(
                f"{label}: record {position} expected_domain {recorded_expected!r} disagrees with its source block"
            )
        if "prompt_reference_count" not in record or "references" not in record:
            raise ValueError(f"{label}: record {position} lacks prompt_reference_count/references")
        references = record["references"]
        if not isinstance(references, list) or any(not isinstance(item, Mapping) for item in references):
            raise ValueError(f"{label}: record {position}.references must be a list of dictionaries")
        prompt_count = _integer(
            record["prompt_reference_count"],
            f"{label}.records[{position}].prompt_reference_count",
        )
        if prompt_count != len(references):
            raise ValueError(
                f"{label}: record {position} captured {len(references)} references but reports {prompt_count}"
            )
        records_by_block[block_index].append(record)

    empty_blocks = [str(index) for index, records in records_by_block.items() if not records]
    if empty_blocks:
        raise ValueError(f"{label}: no output records fall in block(s) {', '.join(empty_blocks)}")
    config = payload.get("config") or {}
    if not isinstance(config, Mapping):
        raise ValueError(f"{label}: config must be a dictionary")
    return ValidatedSource(
        payload=payload,
        label=label,
        sha256=sha256,
        config=config,
        blocks=tuple(raw_blocks),
        spans=spans,
        records_by_block={index: tuple(records) for index, records in records_by_block.items()},
        chunk_samples=_resolve_chunk_samples(payload, label),
    )


def _is_sample_position_key(key: str) -> bool:
    lowered = str(key).lower()
    if lowered in {"start_sample", "end_sample", "cursor_samples", "last_llm_samples", "boundary_sample"}:
        return True
    if not (lowered.endswith("_sample") or lowered.endswith("_samples")):
        return False
    tokens = set(lowered.removesuffix("_samples").removesuffix("_sample").split("_"))
    return bool(tokens & _POSITION_TOKENS)


def _time_key_for_sample_key(key: str) -> str:
    if key.endswith("_samples"):
        return key[: -len("_samples")] + "_s"
    return key[: -len("_sample")] + "_s"


def _clip_and_rebase_record(
    record: Mapping[str, Any],
    *,
    source_span: Mapping[str, Any],
    output_start_sample: int,
    output_block_index: int,
    output_event_index: int,
    source_run_index: int,
    source_record_index: int,
) -> Dict[str, Any]:
    source_start = int(source_span["start_sample"])
    source_end = int(source_span["end_sample"])
    row = copy.deepcopy(dict(record))
    original_event_index = row.get("event_idx")
    changed_positions: Dict[str, int] = {}
    for key, raw_value in list(row.items()):
        if not _is_sample_position_key(key):
            continue
        value = _integer(raw_value, f"record.{key}")
        clipped = min(source_end, max(source_start, value))
        changed_positions[key] = output_start_sample + clipped - source_start
        row[key] = changed_positions[key]
    row["start_sample"] = changed_positions["start_sample"]
    row["cursor_samples"] = changed_positions["cursor_samples"]
    for sample_key, value in changed_positions.items():
        seconds_key = _time_key_for_sample_key(sample_key)
        if seconds_key in row or seconds_key == "cursor_s":
            row[seconds_key] = round(value / TARGET_SAMPLE_RATE, 3)
    row["cursor_s"] = round(int(row["cursor_samples"]) / TARGET_SAMPLE_RATE, 3)
    row["event_idx"] = output_event_index
    row["block_index"] = output_block_index
    row["expected_domain"] = str(source_span["expected_domain"])
    row["oracle_source"] = {
        "run_index": source_run_index,
        "block_index": int(source_span["block_index"]),
        "record_index": source_record_index,
        "event_idx": original_event_index,
        "start_sample": int(record["start_sample"]),
        "cursor_samples": int(record["cursor_samples"]),
    }
    return row


def _shared_config(sources: Sequence[ValidatedSource]) -> Dict[str, Any]:
    for key in _REQUIRED_SHARED_CONFIG_KEYS:
        missing = [source.label for source in sources if key not in source.config]
        if missing:
            raise ValueError(f"source run(s) missing config.{key}: {', '.join(missing)}")
    for key in _STRICT_SHARED_CONFIG_KEYS:
        present = [(source.label, source.config[key]) for source in sources if key in source.config]
        if present and any(value != present[0][1] for _, value in present[1:]):
            detail = ", ".join(f"{label}={value!r}" for label, value in present)
            raise ValueError(f"source runs disagree on config.{key}: {detail}")
    chunk_samples = sources[0].chunk_samples
    if any(source.chunk_samples != chunk_samples for source in sources[1:]):
        detail = ", ".join(f"{source.label}={source.chunk_samples}" for source in sources)
        raise ValueError(f"source runs disagree on config.chunk_samples: {detail}")

    common_keys = set(sources[0].config)
    for source in sources[1:]:
        common_keys &= set(source.config)
    common: Dict[str, Any] = {}
    for key in sorted(common_keys - _RUN_SPECIFIC_CONFIG_KEYS):
        value = sources[0].config[key]
        if all(source.config[key] == value for source in sources[1:]):
            common[key] = copy.deepcopy(value)
    common.update(
        {
            "schedule": "blockwise_oracle",
            "preset": "blockwise_oracle",
            "composition": "ordered_selected_blocks",
            "sample_rate": TARGET_SAMPLE_RATE,
            "chunk_samples": chunk_samples,
            "chunk_seconds": round(chunk_samples / TARGET_SAMPLE_RATE, 3),
        }
    )
    return common


def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _composition_transitions(
    spans: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    max_switch_events: int,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    for previous, current in zip(spans, spans[1:]):
        if previous["expected_domain"] == current["expected_domain"]:
            continue
        block_records = [row for row in records if int(row["block_index"]) == int(current["block_index"])]
        first_target = next(
            (row for row in block_records if row.get("active_domain") == current["expected_domain"]),
            None,
        )
        first_event = block_records[0] if block_records else None
        latency_events = None
        latency_s = None
        if first_target is not None and first_event is not None:
            latency_events = int(first_target["event_idx"]) - int(first_event["event_idx"]) + 1
            latency_s = max(
                0.0,
                (int(first_target["cursor_samples"]) - int(current["start_sample"])) / TARGET_SAMPLE_RATE,
            )
        transitions.append(
            {
                "from_block_index": int(previous["block_index"]),
                "to_block_index": int(current["block_index"]),
                "from_item_id": str(previous["item_id"]),
                "to_item_id": str(current["item_id"]),
                "from_domain": str(previous["expected_domain"]),
                "to_domain": str(current["expected_domain"]),
                "boundary_sample": int(current["start_sample"]),
                "boundary_s": round(int(current["start_sample"]) / TARGET_SAMPLE_RATE, 3),
                "first_target_active_event": first_target.get("event_idx") if first_target else None,
                "first_target_active_s": first_target.get("cursor_s") if first_target else None,
                "latency_events": latency_events,
                "latency_s": round(latency_s, 3) if latency_s is not None else None,
                "max_switch_events": max_switch_events,
                "passed": bool(latency_events is not None and latency_events <= max_switch_events),
                "composition_boundary": True,
            }
        )
    return transitions


def _summarize(
    spans: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    *,
    chunk_samples: int,
    source_run_count: int,
    max_switch_events: int,
) -> Dict[str, Any]:
    expected_records = [row for row in records if row.get("expected_domain")]
    active_correct = sum(row.get("active_domain") == row.get("expected_domain") for row in expected_records)
    probe_records = [row for row in expected_records if row.get("domain_probe_top_domain")]
    probe_correct = sum(row.get("domain_probe_top_domain") == row.get("expected_domain") for row in probe_records)
    retrieve_values = [float(row["retrieve_s"]) for row in records if row.get("retrieve_s") is not None]
    probe_values = [float(row["domain_probe_s"]) for row in records if row.get("domain_probe_s") is not None]
    references = [len(row.get("references") or []) for row in records]
    active_accuracy = active_correct / len(expected_records) if expected_records else 0.0
    p50_retrieve = _percentile(retrieve_values, 50)
    p95_retrieve = _percentile(retrieve_values, 95)
    p50_probe = _percentile(probe_values, 50)
    p95_probe = _percentile(probe_values, 95)
    return {
        "schedule": "blockwise_oracle",
        "preset": "blockwise_oracle",
        "composition": "ordered_selected_blocks",
        "source_run_count": source_run_count,
        "block_count": len(spans),
        "audio_seconds": round((int(spans[-1]["end_sample"]) if spans else 0) / TARGET_SAMPLE_RATE, 3),
        "event_count": len(records),
        "chunk_samples": chunk_samples,
        "chunk_seconds": round(chunk_samples / TARGET_SAMPLE_RATE, 3),
        "domain_transition_count": len(transitions),
        "composition_boundary_count": max(0, len(spans) - 1),
        "switch_count": 0,
        "max_switch_events": max_switch_events,
        "active_domain_accuracy": round(active_accuracy, 4),
        "steady_state_active_domain_accuracy": round(active_accuracy, 4),
        "steady_state_mismatch_count": len(expected_records) - active_correct,
        "probe_top_accuracy": round(probe_correct / len(probe_records), 4) if probe_records else None,
        "probe_seen_events": len(probe_records),
        "wrong_switch_count": 0,
        "router_text_sources": dict(Counter(str(row.get("router_text_source") or "") for row in records)),
        "active_domains": dict(Counter(str(row.get("active_domain") or "") for row in records)),
        "probe_top_domains": dict(Counter(str(row.get("domain_probe_top_domain") or "") for row in records)),
        "retrieved_references_per_chunk": round(statistics.mean(references), 6) if references else 0.0,
        "retrieve_p50_ms": round(p50_retrieve * 1000.0, 2) if p50_retrieve is not None else None,
        "retrieve_p95_ms": round(p95_retrieve * 1000.0, 2) if p95_retrieve is not None else None,
        "domain_probe_p50_ms": round(p50_probe * 1000.0, 2) if p50_probe is not None else None,
        "domain_probe_p95_ms": round(p95_probe * 1000.0, 2) if p95_probe is not None else None,
        "transition_pass": all(bool(row["passed"]) for row in transitions),
        "regression_pass": bool(expected_records and active_correct == len(expected_records)),
    }


def stitch_selected_blocks(
    selections: Sequence[BlockSelection],
    *,
    max_switch_events: int = 3,
    require_oracle_domain: bool = True,
) -> Dict[str, Any]:
    """Stitch selected blocks in the exact order supplied."""

    if not selections:
        raise ValueError("at least one block selection is required")
    if max_switch_events <= 0:
        raise ValueError("max_switch_events must be positive")

    cache: Dict[Tuple[int, str, str], ValidatedSource] = {}
    ordered: List[Tuple[ValidatedSource, int]] = []
    selected_source_blocks: set[Tuple[str, int]] = set()
    selected_identities: set[Tuple[str, str]] = set()
    for selection in selections:
        cache_key = (id(selection.payload), selection.source_label, selection.source_sha256)
        source = cache.get(cache_key)
        if source is None:
            source = validate_source_payload(
                selection.payload,
                label=selection.source_label,
                sha256=selection.source_sha256,
            )
            cache[cache_key] = source
        block_index = _integer(selection.block_index, f"{selection.source_label}.block_index")
        if not 1 <= block_index <= len(source.blocks):
            raise ValueError(
                f"{selection.source_label}: selected block {block_index} is outside 1..{len(source.blocks)}"
            )
        source_key = (source.sha256 or source.label, block_index)
        if source_key in selected_source_blocks:
            raise ValueError(f"duplicate block selection {source.label}={block_index}")
        selected_source_blocks.add(source_key)
        identity = (str(source.blocks[block_index - 1]["corpus"]), str(source.blocks[block_index - 1]["item_id"]))
        if identity in selected_identities:
            raise ValueError(f"duplicate output block identity {identity!r}")
        selected_identities.add(identity)
        ordered.append((source, block_index))

    unique_sources: List[ValidatedSource] = []
    source_run_index: Dict[Tuple[int, str, str], int] = {}
    for source, _ in ordered:
        key = (id(source.payload), source.label, source.sha256)
        if key not in source_run_index:
            source_run_index[key] = len(unique_sources) + 1
            unique_sources.append(source)
    config = _shared_config(unique_sources)
    config["source_run_count"] = len(unique_sources)
    config["selected_block_count"] = len(ordered)
    config["max_switch_events"] = max_switch_events

    blocks: List[Dict[str, Any]] = []
    spans: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []
    block_sources: List[Dict[str, Any]] = []
    output_cursor = 0
    event_index = 0
    for output_block_index, (source, source_block_index) in enumerate(ordered, start=1):
        source_key = (id(source.payload), source.label, source.sha256)
        run_index = source_run_index[source_key]
        block = copy.deepcopy(dict(source.blocks[source_block_index - 1]))
        source_span = source.spans[source_block_index - 1]
        duration = int(source_span["end_sample"]) - int(source_span["start_sample"])
        span = copy.deepcopy(dict(source_span))
        span.update(
            {
                "block_index": output_block_index,
                "start_sample": output_cursor,
                "end_sample": output_cursor + duration,
                "sample_count": duration,
            }
        )
        if "start_s" in span:
            span["start_s"] = round(output_cursor / TARGET_SAMPLE_RATE, 3)
        if "end_s" in span:
            span["end_s"] = round((output_cursor + duration) / TARGET_SAMPLE_RATE, 3)
        blocks.append(block)
        spans.append(span)
        source_records = source.records_by_block[source_block_index]
        for source_record_index, record in enumerate(source_records, start=1):
            active_domain = str(record.get("active_domain") or "")
            expected_domain = str(source_span["expected_domain"])
            if require_oracle_domain and active_domain != expected_domain:
                raise ValueError(
                    f"{source.label}: block {source_block_index} record {source_record_index} has "
                    f"active_domain={active_domain!r}, expected {expected_domain!r}"
                )
            event_index += 1
            records.append(
                _clip_and_rebase_record(
                    record,
                    source_span=source_span,
                    output_start_sample=output_cursor,
                    output_block_index=output_block_index,
                    output_event_index=event_index,
                    source_run_index=run_index,
                    source_record_index=source_record_index,
                )
            )
        block_sources.append(
            {
                "output_block_index": output_block_index,
                "source_run_index": run_index,
                "source_block_index": source_block_index,
                "source_item_id": str(source_span["item_id"]),
                "source_preset": str(source.config.get("preset") or ""),
                "source_start_sample": int(source_span["start_sample"]),
                "source_end_sample": int(source_span["end_sample"]),
                "output_start_sample": output_cursor,
                "output_end_sample": output_cursor + duration,
                "record_count": len(source_records),
            }
        )
        output_cursor += duration

    medicine_ids = [
        str(block["item_id"]).removeprefix("medicine_")
        for block in blocks
        if str(block.get("corpus") or "") == "medicine"
    ]
    config["medicine_ids"] = medicine_ids
    config["oracle_presets"] = [row["source_preset"] for row in block_sources]
    config["fixed_prompt_k_values"] = sorted(
        {int(row["fixed_prompt_k"]) for row in records if row.get("fixed_prompt_k") is not None}
    )

    transitions = _composition_transitions(
        spans,
        records,
        max_switch_events=max_switch_events,
    )
    source_runs: List[Dict[str, Any]] = []
    source_sessions: List[Dict[str, Any]] = []
    for index, source in enumerate(unique_sources, start=1):
        source_runs.append(
            {
                "source_run_index": index,
                "label": source.label,
                "sha256": source.sha256 or None,
                "config": copy.deepcopy(dict(source.config)),
                "summary": copy.deepcopy(source.payload.get("summary") or {}),
                "session": copy.deepcopy(source.payload.get("session") or {}),
                "block_count": len(source.blocks),
                "record_count": sum(len(rows) for rows in source.records_by_block.values()),
            }
        )
        source_sessions.append(
            {
                "source_run_index": index,
                "session": copy.deepcopy(source.payload.get("session") or {}),
            }
        )
    summary = _summarize(
        spans,
        records,
        transitions,
        chunk_samples=int(config["chunk_samples"]),
        source_run_count=len(unique_sources),
        max_switch_events=max_switch_events,
    )
    payload = {
        "config": config,
        "blocks": blocks,
        "block_spans": spans,
        "summary": summary,
        "domain_transitions": transitions,
        "records": records,
        "session": {
            "session_id": "blockwise-oracle-composite",
            "composed": True,
            "source_sessions": source_sessions,
        },
        "oracle_composition": {
            "source_runs": source_runs,
            "block_sources": block_sources,
        },
    }
    validate_source_payload(payload, label="composed blockwise oracle")
    return payload


def stitch_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    source_labels: Optional[Sequence[str]] = None,
    max_switch_events: int = 3,
    require_oracle_domain: bool = True,
) -> Dict[str, Any]:
    """Expand every block from each payload and stitch them in payload order."""

    labels = list(source_labels or [f"run-{index}" for index in range(1, len(payloads) + 1)])
    if len(labels) != len(payloads):
        raise ValueError("source_labels must have the same length as payloads")
    selections: List[BlockSelection] = []
    for payload, label in zip(payloads, labels):
        blocks = payload.get("blocks") if isinstance(payload, Mapping) else None
        if not isinstance(blocks, list):
            raise ValueError(f"{label}: blocks must be a list")
        selections.extend(
            BlockSelection(payload=payload, block_index=index, source_label=label)
            for index in range(1, len(blocks) + 1)
        )
    return stitch_selected_blocks(
        selections,
        max_switch_events=max_switch_events,
        require_oracle_domain=require_oracle_domain,
    )


def _read_payload(path: Path) -> Tuple[Dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top-level JSON value must be a dictionary")
    return payload, hashlib.sha256(raw).hexdigest()


def parse_block_selector(value: str) -> Tuple[Path, int]:
    path_text, separator, index_text = str(value).rpartition("=")
    if not separator or not path_text or not index_text:
        raise argparse.ArgumentTypeError("block selector must use PATH=BLOCK_INDEX")
    try:
        index = int(index_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("block selector index must be an integer") from exc
    if index <= 0:
        raise argparse.ArgumentTypeError("block selector index must be positive")
    return Path(path_text).expanduser(), index


def _write_json_atomic(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_markdown(payload: Mapping[str, Any], path: Path) -> None:
    summary = payload["summary"]
    lines = [
        "# Blockwise Oracle Composition",
        "",
        "The payload was assembled from fixed-domain runs; composition boundaries are not online router switches.",
        "",
        "## Summary",
        "",
        f"- Blocks: `{summary['block_count']}`",
        f"- Source runs: `{summary['source_run_count']}`",
        f"- Audio seconds: `{summary['audio_seconds']}`",
        f"- Events: `{summary['event_count']}`",
        f"- Active-domain accuracy: `{summary['active_domain_accuracy']}`",
        f"- Retrieved references per chunk: `{summary['retrieved_references_per_chunk']}`",
        "",
        "## Block provenance",
        "",
        "| output block | item | source run | source block | preset | records |",
        "|---:|---|---:|---:|---|---:|",
    ]
    for row in payload["oracle_composition"]["block_sources"]:
        lines.append(
            f"| {row['output_block_index']} | {row['source_item_id']} | {row['source_run_index']} | "
            f"{row['source_block_index']} | {row['source_preset']} | {row['record_count']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Full mixed-run payloads; all blocks are appended in file order",
    )
    parser.add_argument(
        "--block",
        action="append",
        default=[],
        type=parse_block_selector,
        metavar="PATH=INDEX",
        help="Select one block; repeat in the desired output order",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--max-switch-events", type=int, default=3)
    parser.add_argument("--allow-active-domain-mismatch", action="store_true")
    args = parser.parse_args()
    if bool(args.inputs) == bool(args.block):
        parser.error("provide either positional full payloads or one or more --block selectors")

    cache: Dict[Path, Tuple[Dict[str, Any], str]] = {}

    def load(path: Path) -> Tuple[Dict[str, Any], str, Path]:
        resolved = path.expanduser().resolve()
        if resolved not in cache:
            cache[resolved] = _read_payload(resolved)
        payload, digest = cache[resolved]
        return payload, digest, resolved

    selections: List[BlockSelection] = []
    if args.block:
        for path, block_index in args.block:
            payload, digest, resolved = load(path)
            selections.append(
                BlockSelection(
                    payload=payload,
                    block_index=block_index,
                    source_label=str(resolved),
                    source_sha256=digest,
                )
            )
    else:
        for path in args.inputs:
            payload, digest, resolved = load(path)
            blocks = payload.get("blocks")
            if not isinstance(blocks, list):
                raise ValueError(f"{resolved}: blocks must be a list")
            selections.extend(
                BlockSelection(
                    payload=payload,
                    block_index=index,
                    source_label=str(resolved),
                    source_sha256=digest,
                )
                for index in range(1, len(blocks) + 1)
            )
    output = args.out_json.expanduser().resolve()
    if output in cache:
        parser.error("--out-json must not overwrite an input payload")
    payload = stitch_selected_blocks(
        selections,
        max_switch_events=args.max_switch_events,
        require_oracle_domain=not args.allow_active_domain_mismatch,
    )
    _write_json_atomic(payload, output)
    if args.out_md:
        _write_markdown(payload, args.out_md.expanduser().resolve())
    print(
        json.dumps(
            {
                "out_json": str(output),
                "blocks": payload["summary"]["block_count"],
                "events": payload["summary"]["event_count"],
                "audio_seconds": payload["summary"]["audio_seconds"],
                "active_domain_accuracy": payload["summary"]["active_domain_accuracy"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

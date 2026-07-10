#!/usr/bin/env python3
"""Score whole-playlist char-level StreamLAAL from a mixed-run JSON.

This scorer is intentionally independent of the no-RAG SimulEval agent.  It
consumes the partial outputs persisted by ``eval_mixed_audio_switch.py`` and
treats the complete mixed playlist as one speech-to-text instance:

* each emitted target character receives the record's ``cursor_samples`` as
  its source delay;
* each emitted target character receives the record's ``emitted_wall_s`` as
  its raw computation-aware elapsed timestamp;
* the raw elapsed timestamps are converted with the same segment-level update
  used by FBK ``stream_laal_term.py`` v2.2 before computation-aware LAAL;
* LAAL uses ``max(reference_length, hypothesis_length)`` exactly as SimulEval's
  ``LAALScorer`` does.

The metric is a single whole-playlist score, not a talk macro.  It neither
resegments text with mWERSegmenter nor pretends that block-boundary-crossing
decoder chunks can be assigned exactly to individual talks.  Computation-aware
scores are meaningful only when ``emitted_wall_s`` is relative to one stable
client-side origin and all compared runs use the same pacing protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


SCHEMA_VERSION = "mixed_streamlaal.v1"
DEFAULT_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class CharacterLatencySeries:
    prediction: str
    delays_ms: tuple[float, ...]
    raw_elapsed_ms: tuple[float, ...]
    stream_elapsed_ms: tuple[float, ...]
    records_total: int
    records_with_text: int
    records_with_only_whitespace: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(raw)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_text(path: Path) -> str:
    """Load one char-level playlist reference without counting line breaks."""

    lines = path.read_text(encoding="utf-8").splitlines()
    return "".join(line.strip() for line in lines).strip()


def _finite_float(value: Any, *, field: str, record_index: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"record {record_index} has invalid {field}={value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"record {record_index} has non-finite {field}={value!r}")
    return result


def _playlist_geometry(
    payload: Mapping[str, Any],
    *,
    expected_block_count: int = 0,
) -> tuple[list[Dict[str, Any]], int]:
    raw_spans = payload.get("block_spans") or []
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("mixed-run payload has no block_spans")
    spans = sorted(
        (dict(item) for item in raw_spans if isinstance(item, Mapping)),
        key=lambda item: int(item.get("block_index") or 0),
    )
    if len(spans) != len(raw_spans):
        raise ValueError("every block_spans entry must be an object")
    if expected_block_count > 0 and len(spans) != expected_block_count:
        raise ValueError(
            f"playlist has {len(spans)} blocks, expected {expected_block_count}"
        )

    previous_end = 0
    normalized: list[Dict[str, Any]] = []
    for position, span in enumerate(spans, start=1):
        block_index = int(span.get("block_index") or 0)
        start = int(span.get("start_sample") or 0)
        end = int(span.get("end_sample") or 0)
        declared_count = int(span.get("sample_count") or 0)
        if block_index != position:
            raise ValueError(
                f"block_spans must use consecutive 1-based block_index values; "
                f"position {position} has {block_index}"
            )
        if start != previous_end:
            raise ValueError(
                f"playlist spans are not contiguous at block {block_index}: "
                f"start={start}, previous_end={previous_end}"
            )
        if end <= start or declared_count != end - start:
            raise ValueError(
                f"block {block_index} has invalid sample geometry "
                f"start={start}, end={end}, sample_count={declared_count}"
            )
        normalized.append(
            {
                "block_index": block_index,
                "item_id": str(span.get("item_id") or ""),
                "corpus": str(span.get("corpus") or ""),
                "expected_domain": str(span.get("expected_domain") or ""),
                "start_sample": start,
                "end_sample": end,
                "sample_count": declared_count,
            }
        )
        previous_end = end
    return normalized, previous_end


def fbk_stream_elapsed(
    delays_ms: Sequence[float],
    elapsed_ms: Sequence[float],
    *,
    sentence_start_ms: float = 0.0,
) -> list[float]:
    """Mirror FBK v2.2 ``SegmentLevelDelayElapsed`` for one sentence."""

    if len(delays_ms) != len(elapsed_ms):
        raise ValueError("delays and elapsed timestamps must have equal lengths")
    if sentence_start_ms < 0.0 or not math.isfinite(sentence_start_ms):
        raise ValueError("sentence_start_ms must be finite and non-negative")

    stream_elapsed: list[float] = []
    previous_delay: float | None = None
    previous_elapsed: float | None = None
    previous_stream_delay: float | None = None
    previous_stream_elapsed: float | None = None
    for delay, elapsed in zip(delays_ms, elapsed_ms):
        delay = float(delay)
        elapsed = float(elapsed)
        if previous_elapsed is None:
            current_stream_elapsed = elapsed
        elif elapsed == previous_elapsed:
            assert previous_stream_elapsed is not None
            assert previous_stream_delay is not None
            current_stream_elapsed = previous_stream_elapsed - previous_stream_delay + delay
        else:
            assert previous_delay is not None
            current_stream_elapsed = elapsed - previous_elapsed + previous_delay

        current_stream_delay = max(0.0, delay - sentence_start_ms)
        current_stream_elapsed = max(0.0, current_stream_elapsed - sentence_start_ms)
        stream_elapsed.append(current_stream_elapsed)
        previous_delay = delay
        previous_elapsed = elapsed
        previous_stream_delay = current_stream_delay
        previous_stream_elapsed = current_stream_elapsed
    return stream_elapsed


def build_character_latency_series(
    payload: Mapping[str, Any],
    *,
    source_length_samples: int,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> CharacterLatencySeries:
    records = payload.get("records") or []
    if not isinstance(records, list) or not records:
        raise ValueError("mixed-run payload has no records")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    prediction_parts: list[str] = []
    delays_ms: list[float] = []
    raw_elapsed_ms: list[float] = []
    previous_cursor = -1
    previous_wall_s = -1.0
    records_with_text = 0
    records_with_only_whitespace = 0

    for record_index, raw_record in enumerate(records, start=1):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"record {record_index} is not an object")
        if "cursor_samples" not in raw_record:
            raise ValueError(f"record {record_index} lacks cursor_samples")
        if "emitted_wall_s" not in raw_record:
            raise ValueError(f"record {record_index} lacks emitted_wall_s")
        if "text" not in raw_record:
            raise ValueError(
                f"record {record_index} lacks full text; text_preview is insufficient for StreamLAAL"
            )

        cursor = int(raw_record["cursor_samples"])
        wall_s = _finite_float(
            raw_record["emitted_wall_s"],
            field="emitted_wall_s",
            record_index=record_index,
        )
        if cursor < 0 or cursor < previous_cursor:
            raise ValueError(
                f"record {record_index} has non-monotonic cursor_samples={cursor}; "
                f"previous={previous_cursor}"
            )
        if cursor > source_length_samples:
            raise ValueError(
                f"record {record_index} cursor_samples={cursor} exceeds playlist "
                f"length {source_length_samples}"
            )
        if wall_s < 0.0 or wall_s < previous_wall_s:
            raise ValueError(
                f"record {record_index} has non-monotonic emitted_wall_s={wall_s}; "
                f"previous={previous_wall_s}"
            )
        text = raw_record["text"]
        if not isinstance(text, str):
            raise ValueError(f"record {record_index}.text must be a string")

        prediction_parts.append(text)
        if text:
            records_with_text += 1
        if text and not text.strip():
            records_with_only_whitespace += 1
        character_count = len(text)
        delay_ms = 1000.0 * cursor / sample_rate
        elapsed_ms = 1000.0 * wall_s
        delays_ms.extend([delay_ms] * character_count)
        raw_elapsed_ms.extend([elapsed_ms] * character_count)
        previous_cursor = cursor
        previous_wall_s = wall_s

    raw_prediction = "".join(prediction_parts)
    left = len(raw_prediction) - len(raw_prediction.lstrip())
    right = len(raw_prediction.rstrip())
    prediction = raw_prediction[left:right]
    delays_ms = delays_ms[left:right]
    raw_elapsed_ms = raw_elapsed_ms[left:right]
    if not prediction:
        raise ValueError("mixed-run payload has no non-whitespace emitted characters")
    if len(prediction) != len(delays_ms) or len(prediction) != len(raw_elapsed_ms):
        raise RuntimeError("prediction/timestamp alignment failed")

    stream_elapsed_ms = fbk_stream_elapsed(delays_ms, raw_elapsed_ms)
    return CharacterLatencySeries(
        prediction=prediction,
        delays_ms=tuple(delays_ms),
        raw_elapsed_ms=tuple(raw_elapsed_ms),
        stream_elapsed_ms=tuple(stream_elapsed_ms),
        records_total=len(records),
        records_with_text=records_with_text,
        records_with_only_whitespace=records_with_only_whitespace,
    )


def compute_laal(
    timestamps_ms: Sequence[float],
    *,
    source_length_ms: float,
    reference_length: int,
) -> Dict[str, Any]:
    """Compute SimulEval ``LAALScorer`` for one text-output instance."""

    if not timestamps_ms:
        raise ValueError("LAAL requires at least one target timestamp")
    if source_length_ms <= 0.0 or not math.isfinite(source_length_ms):
        raise ValueError("source_length_ms must be finite and positive")
    if reference_length <= 0:
        raise ValueError("reference_length must be positive")
    values = [float(value) for value in timestamps_ms]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("LAAL timestamps must be finite and non-negative")

    hypothesis_length = len(values)
    adaptive_target_length = max(hypothesis_length, int(reference_length))
    ideal_step_ms = source_length_ms / adaptive_target_length
    if values[0] > source_length_ms:
        return {
            "score_ms": values[0],
            "tau": 1,
            "reached_source_end": True,
            "hypothesis_length": hypothesis_length,
            "reference_length": int(reference_length),
            "adaptive_target_length": adaptive_target_length,
            "ideal_step_ms": ideal_step_ms,
        }

    total = 0.0
    tau = 0
    reached_source_end = False
    for target_index, timestamp in enumerate(values):
        total += timestamp - target_index * ideal_step_ms
        tau = target_index + 1
        if timestamp >= source_length_ms:
            reached_source_end = True
            break
    return {
        "score_ms": total / tau,
        "tau": tau,
        "reached_source_end": reached_source_end,
        "hypothesis_length": hypothesis_length,
        "reference_length": int(reference_length),
        "adaptive_target_length": adaptive_target_length,
        "ideal_step_ms": ideal_step_ms,
    }


def score_payload(
    payload: Mapping[str, Any],
    *,
    reference_text: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    expected_block_count: int = 0,
) -> Dict[str, Any]:
    reference = "".join(line.strip() for line in str(reference_text).splitlines()).strip()
    if not reference:
        raise ValueError("whole-playlist reference text is empty")
    spans, source_length_samples = _playlist_geometry(
        payload,
        expected_block_count=expected_block_count,
    )
    series = build_character_latency_series(
        payload,
        source_length_samples=source_length_samples,
        sample_rate=sample_rate,
    )
    source_length_ms = 1000.0 * source_length_samples / sample_rate
    reference_length = len(reference)
    standard = compute_laal(
        series.delays_ms,
        source_length_ms=source_length_ms,
        reference_length=reference_length,
    )
    computation_aware = compute_laal(
        series.stream_elapsed_ms,
        source_length_ms=source_length_ms,
        reference_length=reference_length,
    )
    last_cursor_samples = int((payload.get("records") or [])[-1]["cursor_samples"])
    event_signature = [
        {
            "cursor_samples": int(record["cursor_samples"]),
            "text": str(record["text"]),
            "emitted_wall_s": float(record["emitted_wall_s"]),
        }
        for record in payload.get("records") or []
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "instance_definition": "one complete mixed playlist",
            "latency_unit": "Unicode character",
            "timestamp_unit": "milliseconds",
            "source_delay": "record.cursor_samples / sample_rate",
            "raw_elapsed": "record.emitted_wall_s from one client-relative origin",
            "computation_aware_transform": "FBK stream_laal_term.py v2.2 SegmentLevelDelayElapsed",
            "laal_definition": (
                "mean over t<=tau of timestamp[t] - t*source_length/"
                "max(reference_chars,hypothesis_chars)"
            ),
            "tau_definition": "first target character with timestamp >= source length; otherwise all characters",
            "reference_line_policy": "strip each line and concatenate without line-break characters",
            "prediction_policy": "concatenate record.text exactly, then strip only global outer whitespace",
            "resegmentation": False,
            "talk_macro": False,
            "boundary_note": (
                "decoder chunks may cross block boundaries; the playlist is intentionally scored as one instance"
            ),
        },
        "playlist": {
            "block_count": len(spans),
            "blocks": spans,
            "sha256": _stable_sha256(spans),
            "sample_rate": int(sample_rate),
            "source_length_samples": source_length_samples,
            "source_length_ms": source_length_ms,
            "last_cursor_samples": last_cursor_samples,
            "tail_gap_samples": source_length_samples - last_cursor_samples,
        },
        "text": {
            "reference_chars": reference_length,
            "reference_sha256": _sha256_bytes(reference.encode("utf-8")),
            "hypothesis_chars": len(series.prediction),
            "hypothesis_sha256": _sha256_bytes(series.prediction.encode("utf-8")),
        },
        "records": {
            "total": series.records_total,
            "with_text": series.records_with_text,
            "with_only_whitespace": series.records_with_only_whitespace,
            "event_signature_sha256": _stable_sha256(event_signature),
            "first_cursor_ms": series.delays_ms[0],
            "last_cursor_ms": series.delays_ms[-1],
            "first_raw_elapsed_ms": series.raw_elapsed_ms[0],
            "last_raw_elapsed_ms": series.raw_elapsed_ms[-1],
            "first_stream_elapsed_ms": series.stream_elapsed_ms[0],
            "last_stream_elapsed_ms": series.stream_elapsed_ms[-1],
        },
        "metrics": {
            "stream_laal_ms": standard["score_ms"],
            "stream_laal_s": standard["score_ms"] / 1000.0,
            "stream_laal_ca_ms": computation_aware["score_ms"],
            "stream_laal_ca_s": computation_aware["score_ms"] / 1000.0,
            "standard_diagnostics": standard,
            "computation_aware_diagnostics": computation_aware,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    metrics = report["metrics"]
    playlist = report["playlist"]
    text = report["text"]
    return "\n".join(
        [
            "# Mixed-run StreamLAAL",
            "",
            "The complete mixed playlist is scored as one char-level instance; no talk-level resegmentation is applied.",
            "",
            f"- Blocks: `{playlist['block_count']}`",
            f"- Source duration: `{playlist['source_length_ms'] / 1000.0:.3f}s`",
            f"- Reference / hypothesis characters: `{text['reference_chars']} / {text['hypothesis_chars']}`",
            f"- StreamLAAL: `{metrics['stream_laal_ms']:.3f}ms`",
            f"- StreamLAAL-CA: `{metrics['stream_laal_ca_ms']:.3f}ms`",
            f"- Tail gap: `{playlist['tail_gap_samples']}` samples",
            "",
            "StreamLAAL-CA uses client-side `emitted_wall_s`; comparisons require the same wall-clock origin and pacing protocol.",
            "",
        ]
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--reference-text-file", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--expected-block-count", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-markdown", type=Path)
    args = parser.parse_args(argv)
    if args.sample_rate <= 0:
        parser.error("--sample-rate must be positive")
    if args.expected_block_count < 0:
        parser.error("--expected-block-count must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.run_json.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("run JSON root must be an object")
    reference_text = load_reference_text(args.reference_text_file)
    report = score_payload(
        payload,
        reference_text=reference_text,
        sample_rate=args.sample_rate,
        expected_block_count=args.expected_block_count,
    )
    report["inputs"] = {
        "run_json": str(args.run_json.resolve()),
        "run_json_sha256": file_sha256(args.run_json),
        "reference_text_file": str(args.reference_text_file.resolve()),
        "reference_text_file_sha256": file_sha256(args.reference_text_file),
    }
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(output, end="")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output, encoding="utf-8")
    if args.out_markdown:
        args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.out_markdown.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

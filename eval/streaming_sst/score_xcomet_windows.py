#!/usr/bin/env python3
"""Prepare aligned windows and score paired streaming outputs with xCOMET-lite."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


TARGET_SAMPLE_RATE = 16000
DEFAULT_ACL_ROOT = "/mnt/taurus/data2/jiaxuanluo/rasst_eval/acl6060_zh_segments"
DEFAULT_ACL_SOURCE_TEXT = "/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/acl_zh/source_text.txt"
DEFAULT_ACL_REFERENCE_TEXT = "/mnt/taurus/data2/jiaxuanluo/RASST/data/main_result/inputs/acl_zh/ref.txt"
DEFAULT_MEDICINE_INPUT_DIR = "/mnt/data2/jiaxuanluo/RASST/data/main_result/inputs/medicine_zh"


@dataclass(frozen=True)
class ReferenceSegment:
    start_s: float
    end_s: float
    source: str
    reference: str


def normalise_text(text: str) -> str:
    clean = re.sub(r"</?t>", "", str(text or ""))
    clean = re.sub(r"<[^>]+>", "", clean)
    return re.sub(r"\s+", " ", clean).strip()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def load_acl_meta(acl_root: str) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(Path(acl_root) / "segments.meta.jsonl")
    return {str(row["seg_wav"]): row for row in rows}


def acl_reference_segments(
    block: dict[str, Any],
    block_duration_s: float,
    *,
    acl_meta: dict[str, dict[str, Any]],
    source_lines: Sequence[str],
    reference_lines: Sequence[str],
) -> list[ReferenceSegment]:
    segments: list[ReferenceSegment] = []
    cursor_s = 0.0
    for wav_path in block.get("wav_paths") or []:
        if cursor_s >= block_duration_s:
            break
        meta = acl_meta.get(str(wav_path))
        if meta is None:
            raise ValueError(f"missing ACL metadata for {wav_path}")
        index = int(meta.get("index") if meta.get("index") is not None else meta.get("orig_index", -1))
        if not 0 <= index < len(source_lines) or not 0 <= index < len(reference_lines):
            raise ValueError(f"ACL line index out of range: {index}")
        duration_s = float(meta.get("seg_duration") or meta.get("duration") or 0.0)
        end_s = min(block_duration_s, cursor_s + duration_s)
        if end_s > cursor_s:
            segments.append(
                ReferenceSegment(
                    start_s=cursor_s,
                    end_s=end_s,
                    source=normalise_text(source_lines[index]),
                    reference=normalise_text(reference_lines[index]),
                )
            )
        cursor_s = end_s
    return segments


def medicine_reference_segments(
    block: dict[str, Any],
    block_duration_s: float,
    *,
    medicine_input_dir: str,
) -> list[ReferenceSegment]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is present on eval hosts
        raise RuntimeError("PyYAML is required to read medicine timestamps") from exc

    medicine_id = str(block["item_id"]).removeprefix("medicine_")
    root = Path(medicine_input_dir)
    audio_rows = yaml.safe_load((root / f"medicine.audio__medicine_{medicine_id}.yaml").read_text(encoding="utf-8"))
    source_lines = (root / f"medicine.source_text.en__medicine_{medicine_id}.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    reference_lines = (root / f"medicine.ref.zh__medicine_{medicine_id}.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    if not (len(audio_rows) == len(source_lines) == len(reference_lines)):
        raise ValueError(
            f"medicine alignment mismatch for {medicine_id}: "
            f"audio={len(audio_rows)} src={len(source_lines)} ref={len(reference_lines)}"
        )
    segments: list[ReferenceSegment] = []
    for audio, source, reference in zip(audio_rows, source_lines, reference_lines):
        start_s = float(audio.get("offset") or 0.0)
        if start_s >= block_duration_s:
            break
        end_s = min(block_duration_s, start_s + float(audio.get("duration") or 0.0))
        if end_s > start_s:
            segments.append(
                ReferenceSegment(
                    start_s=start_s,
                    end_s=end_s,
                    source=normalise_text(source),
                    reference=normalise_text(reference),
                )
            )
    return segments


def group_reference_segments(
    segments: Sequence[ReferenceSegment],
    *,
    block_duration_s: float,
    target_window_s: float,
) -> list[dict[str, Any]]:
    if not segments:
        return []
    groups: list[list[ReferenceSegment]] = []
    current: list[ReferenceSegment] = []
    group_start_s = 0.0
    for segment in segments:
        if not current:
            group_start_s = segment.start_s
        current.append(segment)
        if segment.end_s - group_start_s >= target_window_s:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    windows: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        start_s = 0.0 if index == 0 else float(group[0].start_s)
        end_s = (
            float(groups[index + 1][0].start_s)
            if index + 1 < len(groups)
            else float(block_duration_s)
        )
        windows.append(
            {
                "local_start_s": start_s,
                "local_end_s": max(start_s, end_s),
                "source": " ".join(item.source for item in group if item.source),
                "reference": "".join(item.reference for item in group if item.reference),
                "reference_segment_count": len(group),
            }
        )
    return windows


def payload_signature(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    spans = {int(span["block_index"]): span for span in payload.get("block_spans") or []}
    signature: list[tuple[Any, ...]] = []
    for block_index, block in enumerate(payload.get("blocks") or [], start=1):
        span = spans[block_index]
        signature.append(
            (
                block_index,
                str(block.get("item_id")),
                str(block.get("corpus")),
                str(block.get("expected_domain")),
                int(span["sample_count"]),
            )
        )
    return signature


def records_by_block(payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    spans = sorted(payload.get("block_spans") or [], key=lambda row: int(row["block_index"]))
    output: dict[int, list[dict[str, Any]]] = {int(span["block_index"]): [] for span in spans}
    span_index = 0
    for record in payload.get("records") or []:
        cursor = int(record.get("cursor_samples") or 0)
        while span_index + 1 < len(spans) and cursor > int(spans[span_index]["end_sample"]):
            span_index += 1
        span = spans[span_index]
        if int(span["start_sample"]) < cursor <= int(span["end_sample"]):
            output[int(span["block_index"])].append(record)
    return output


def hypothesis_for_window(
    records: Sequence[dict[str, Any]],
    *,
    block_start_sample: int,
    local_start_s: float,
    local_end_s: float,
) -> tuple[str, int]:
    parts: list[str] = []
    event_count = 0
    for record in records:
        local_cursor_s = (int(record.get("cursor_samples") or 0) - block_start_sample) / TARGET_SAMPLE_RATE
        if local_start_s < local_cursor_s <= local_end_s:
            text = normalise_text(record.get("text") or record.get("text_preview") or "")
            if text:
                parts.append(text)
            event_count += 1
    return "".join(parts), event_count


def prepare_windows(
    auto_payload: dict[str, Any],
    merged_payload: dict[str, Any],
    *,
    acl_root: str,
    acl_source_text: str,
    acl_reference_text: str,
    medicine_input_dir: str,
    target_window_s: float,
) -> list[dict[str, Any]]:
    auto_signature = payload_signature(auto_payload)
    merged_signature = payload_signature(merged_payload)
    if auto_signature != merged_signature:
        raise ValueError("AutoTerm and merged runs do not share the same block playlist")

    acl_meta = load_acl_meta(acl_root)
    acl_source_lines = Path(acl_source_text).read_text(encoding="utf-8").splitlines()
    acl_reference_lines = Path(acl_reference_text).read_text(encoding="utf-8").splitlines()
    auto_records = records_by_block(auto_payload)
    merged_records = records_by_block(merged_payload)
    spans = {int(span["block_index"]): span for span in auto_payload.get("block_spans") or []}

    rows: list[dict[str, Any]] = []
    for block_index, block in enumerate(auto_payload.get("blocks") or [], start=1):
        span = spans[block_index]
        block_duration_s = int(span["sample_count"]) / TARGET_SAMPLE_RATE
        if block.get("corpus") == "acl":
            reference_segments = acl_reference_segments(
                block,
                block_duration_s,
                acl_meta=acl_meta,
                source_lines=acl_source_lines,
                reference_lines=acl_reference_lines,
            )
        elif block.get("corpus") == "medicine":
            reference_segments = medicine_reference_segments(
                block,
                block_duration_s,
                medicine_input_dir=medicine_input_dir,
            )
        else:
            raise ValueError(f"unsupported corpus: {block.get('corpus')}")

        grouped = group_reference_segments(
            reference_segments,
            block_duration_s=block_duration_s,
            target_window_s=target_window_s,
        )
        for window_index, window in enumerate(grouped, start=1):
            auto_hypothesis, auto_event_count = hypothesis_for_window(
                auto_records[block_index],
                block_start_sample=int(span["start_sample"]),
                local_start_s=float(window["local_start_s"]),
                local_end_s=float(window["local_end_s"]),
            )
            merged_hypothesis, merged_event_count = hypothesis_for_window(
                merged_records[block_index],
                block_start_sample=int(span["start_sample"]),
                local_start_s=float(window["local_start_s"]),
                local_end_s=float(window["local_end_s"]),
            )
            rows.append(
                {
                    "unit_id": f"b{block_index:02d}_w{window_index:03d}",
                    "block_index": block_index,
                    "window_index": window_index,
                    "item_id": str(block.get("item_id")),
                    "corpus": str(block.get("corpus")),
                    "domain": str(block.get("expected_domain")),
                    **window,
                    "source": normalise_text(window["source"]),
                    "reference": normalise_text(window["reference"]),
                    "auto_hypothesis": auto_hypothesis,
                    "merged_hypothesis": merged_hypothesis,
                    "auto_event_count": auto_event_count,
                    "merged_event_count": merged_event_count,
                }
            )
    return rows


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = float(sum(weights))
    return float(sum(value * weight for value, weight in zip(values, weights)) / total) if total else math.nan


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def paired_bootstrap_ci(
    deltas: Sequence[float],
    block_ids: Sequence[int],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    by_block: dict[int, list[float]] = defaultdict(list)
    for delta, block_id in zip(deltas, block_ids):
        by_block[int(block_id)].append(float(delta))
    block_means = [mean(values) for _, values in sorted(by_block.items())]
    rng = random.Random(seed)
    draws = [
        mean([block_means[rng.randrange(len(block_means))] for _ in block_means])
        for _ in range(samples)
    ]
    return {
        "estimate": mean(block_means),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
    }


def exact_sign_flip_pvalue(deltas: Sequence[float]) -> float | None:
    if not deltas or len(deltas) > 20:
        return None
    observed = abs(mean(deltas))
    extreme = 0
    total = 1 << len(deltas)
    for mask in range(total):
        permuted = [value if mask & (1 << index) else -value for index, value in enumerate(deltas)]
        if abs(mean(permuted)) >= observed - 1e-12:
            extreme += 1
    return extreme / total


def metric_summary(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    auto_values = [float(row[f"auto_{metric}"]) for row in rows]
    merged_values = [float(row[f"merged_{metric}"]) for row in rows]
    deltas = [auto - merged for auto, merged in zip(auto_values, merged_values)]
    weights = [max(1, len(str(row.get("reference") or ""))) for row in rows]
    by_block: dict[int, list[float]] = defaultdict(list)
    for row, delta in zip(rows, deltas):
        by_block[int(row["block_index"])].append(delta)
    block_deltas = [mean(values) for _, values in sorted(by_block.items())]
    auto_talk_macro = mean(
        [
            mean(
                [
                    float(row[f"auto_{metric}"])
                    for row in rows
                    if int(row["block_index"]) == block
                ]
            )
            for block in sorted(by_block)
        ]
    )
    merged_talk_macro = mean(
        [
            mean(
                [
                    float(row[f"merged_{metric}"])
                    for row in rows
                    if int(row["block_index"]) == block
                ]
            )
            for block in sorted(by_block)
        ]
    )
    return {
        "segments": len(rows),
        "auto_mean": mean(auto_values),
        "merged_mean": mean(merged_values),
        "delta_mean": mean(deltas),
        "auto_reference_char_weighted": weighted_mean(auto_values, weights),
        "merged_reference_char_weighted": weighted_mean(merged_values, weights),
        "delta_reference_char_weighted": weighted_mean(deltas, weights),
        "auto_talk_macro": auto_talk_macro,
        "merged_talk_macro": merged_talk_macro,
        "delta_talk_macro": auto_talk_macro - merged_talk_macro,
        "win_tie_loss": {
            "auto": sum(delta > 1e-9 for delta in deltas),
            "tie": sum(abs(delta) <= 1e-9 for delta in deltas),
            "merged": sum(delta < -1e-9 for delta in deltas),
        },
        "talk_sign_flip_p_two_sided": exact_sign_flip_pvalue(block_deltas),
    }


def score_windows(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    model_revision: str,
    local_files_only: bool,
    xcomet_code_dir: str,
    batch_size: int,
    gpus: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    max_combined_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if xcomet_code_dir:
        sys.path.insert(0, str(Path(xcomet_code_dir).expanduser()))
    from sacrebleu.metrics import CHRF
    from xcomet.deberta_encoder import XCOMETLite

    model = XCOMETLite().from_pretrained(
        model_name,
        revision=model_revision or None,
        local_files_only=local_files_only,
    )
    data: list[dict[str, str]] = []
    eligible_indices: list[int] = []
    tokenized_rows: list[dict[str, Any]] = []
    tokenizer = model.encoder.tokenizer
    for index, row in enumerate(rows):
        tokenized = dict(row)
        src_tokens = len(tokenizer(str(row["source"]), add_special_tokens=False)["input_ids"])
        ref_tokens = len(tokenizer(str(row["reference"]), add_special_tokens=False)["input_ids"])
        tokenized["source_tokens"] = src_tokens
        tokenized["reference_tokens"] = ref_tokens
        combined_lengths: list[int] = []
        for system in ("auto", "merged"):
            mt_tokens = len(
                tokenizer(str(row[f"{system}_hypothesis"]), add_special_tokens=False)["input_ids"]
            )
            combined_tokens = mt_tokens + src_tokens + ref_tokens + 4
            tokenized[f"{system}_hypothesis_tokens"] = mt_tokens
            tokenized[f"{system}_combined_tokens"] = combined_tokens
            combined_lengths.append(combined_tokens)
        tokenized["xcomet_eligible"] = max(combined_lengths) <= max_combined_tokens
        tokenized_rows.append(tokenized)
        if not tokenized["xcomet_eligible"]:
            continue
        eligible_indices.append(index)
        for system in ("auto", "merged"):
            data.append(
                {
                    "src": str(row["source"]),
                    "mt": str(row[f"{system}_hypothesis"]),
                    "ref": str(row["reference"]),
                }
            )
    prediction = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = [float(score) for score in prediction.scores]
    if len(scores) != 2 * len(eligible_indices):
        raise ValueError(f"expected {2 * len(eligible_indices)} xCOMET scores, got {len(scores)}")
    error_spans = list(getattr(prediction.metadata, "error_spans", [[] for _ in scores]))

    chrf = CHRF(char_order=6, word_order=0, beta=2)
    scored_rows: list[dict[str, Any]] = []
    eligible_position = {row_index: position for position, row_index in enumerate(eligible_indices)}
    for index, row in enumerate(tokenized_rows):
        scored = dict(row)
        if index in eligible_position:
            position = eligible_position[index]
            scored["auto_xcomet_lite"] = scores[2 * position]
            scored["merged_xcomet_lite"] = scores[2 * position + 1]
            scored["auto_xcomet_error_spans"] = error_spans[2 * position]
            scored["merged_xcomet_error_spans"] = error_spans[2 * position + 1]
        else:
            scored["auto_xcomet_lite"] = None
            scored["merged_xcomet_lite"] = None
            scored["auto_xcomet_error_spans"] = []
            scored["merged_xcomet_error_spans"] = []
        scored["auto_chrf_pp"] = float(
            chrf.sentence_score(str(row["auto_hypothesis"]), [str(row["reference"])]).score
        )
        scored["merged_chrf_pp"] = float(
            chrf.sentence_score(str(row["merged_hypothesis"]), [str(row["reference"])]).score
        )
        scored_rows.append(scored)

    metrics: dict[str, Any] = {}
    for metric in ("xcomet_lite", "chrf_pp"):
        metric_rows = [
            row
            for row in scored_rows
            if row[f"auto_{metric}"] is not None and row[f"merged_{metric}"] is not None
        ]
        summary = metric_summary(metric_rows, metric)
        deltas = [
            float(row[f"auto_{metric}"]) - float(row[f"merged_{metric}"])
            for row in metric_rows
        ]
        summary["talk_bootstrap"] = paired_bootstrap_ci(
            deltas,
            [int(row["block_index"]) for row in metric_rows],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        summary["by_domain"] = {
            domain: metric_summary([row for row in metric_rows if row["domain"] == domain], metric)
            for domain in sorted({str(row["domain"]) for row in metric_rows})
        }
        summary["by_talk"] = {
            item_id: metric_summary([row for row in metric_rows if row["item_id"] == item_id], metric)
            for item_id in dict.fromkeys(str(row["item_id"]) for row in metric_rows)
        }
        metrics[metric] = summary

    severity_counts: dict[str, dict[str, int]] = {
        "auto": defaultdict(int),
        "merged": defaultdict(int),
    }
    output_chars = {"auto": 0, "merged": 0}
    for row in scored_rows:
        if not row["xcomet_eligible"]:
            continue
        for system in ("auto", "merged"):
            output_chars[system] += len(str(row[f"{system}_hypothesis"]))
            for span in row[f"{system}_xcomet_error_spans"]:
                severity_counts[system][str(span.get("severity") or "unknown")] += 1
    error_summary = {}
    for system in ("auto", "merged"):
        counts = dict(severity_counts[system])
        error_summary[system] = {
            "output_chars": output_chars[system],
            "counts": counts,
            "major_plus_critical_per_1k_chars": (
                1000.0 * (counts.get("major", 0) + counts.get("critical", 0)) / output_chars[system]
                if output_chars[system]
                else None
            ),
        }

    return scored_rows, {
        "model": model_name,
        "model_revision": model_revision or None,
        "mode": "reference_based_source_hypothesis_reference",
        "segments": len(scored_rows),
        "xcomet_eligible_segments": len(eligible_indices),
        "xcomet_excluded_overlength": len(scored_rows) - len(eligible_indices),
        "max_combined_tokens": max_combined_tokens,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "xcomet_error_spans": error_summary,
        "metrics": metrics,
    }


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# AutoTerm vs merged glossary: aligned-window quality",
        "",
        f"- Model: `{summary['model']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Aligned windows: {summary['segments']} "
        f"({summary['xcomet_eligible_segments']} xCOMET-eligible; "
        f"{summary['xcomet_excluded_overlength']} overlength excluded)",
        "",
        "| metric | AutoTerm | merged | delta | talk-bootstrap 95% CI | talk sign-flip p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("xCOMET-lite", "xcomet_lite"), ("chrF2", "chrf_pp")):
        metric = summary["metrics"][key]
        bootstrap = metric["talk_bootstrap"]
        lines.append(
            f"| {label} | {metric['auto_talk_macro']:.6f} | "
            f"{metric['merged_talk_macro']:.6f} | {metric['delta_talk_macro']:+.6f} | "
            f"[{bootstrap['ci95_low']:+.6f}, "
            f"{bootstrap['ci95_high']:+.6f}] | {metric['talk_sign_flip_p_two_sided']:.6f} |"
        )
    lines.extend(["", "## By domain", ""])
    for label, key in (("xCOMET-lite", "xcomet_lite"), ("chrF2", "chrf_pp")):
        lines.extend(
            [
                f"### {label}",
                "",
                "| domain | AutoTerm | merged | delta | windows |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for domain, metric in summary["metrics"][key]["by_domain"].items():
            lines.append(
                f"| {domain} | {metric['auto_mean']:.6f} | {metric['merged_mean']:.6f} | "
                f"{metric['delta_mean']:+.6f} | {metric['segments']} |"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows-jsonl", required=True)
    parser.add_argument("--auto-json", default="")
    parser.add_argument("--merged-json", default="")
    parser.add_argument("--acl-root", default=DEFAULT_ACL_ROOT)
    parser.add_argument("--acl-source-text", default=DEFAULT_ACL_SOURCE_TEXT)
    parser.add_argument("--acl-reference-text", default=DEFAULT_ACL_REFERENCE_TEXT)
    parser.add_argument("--medicine-input-dir", default=DEFAULT_MEDICINE_INPUT_DIR)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model-name", default="myyycroft/XCOMET-lite")
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--xcomet-code-dir", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--max-combined-tokens", type=int, default=480)
    parser.add_argument("--out-scored-jsonl", default="")
    parser.add_argument("--out-summary-json", default="")
    parser.add_argument("--out-md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows_path = Path(args.windows_jsonl)
    if args.auto_json or args.merged_json:
        if not args.auto_json or not args.merged_json:
            raise SystemExit("--auto-json and --merged-json must be provided together")
        rows = prepare_windows(
            read_json(args.auto_json),
            read_json(args.merged_json),
            acl_root=args.acl_root,
            acl_source_text=args.acl_source_text,
            acl_reference_text=args.acl_reference_text,
            medicine_input_dir=args.medicine_input_dir,
            target_window_s=args.window_sec,
        )
        if not rows:
            raise SystemExit("no aligned windows were prepared")
        write_jsonl(windows_path, rows)
        print(json.dumps({"prepared_windows": len(rows), "output": str(windows_path)}, indent=2))
    else:
        rows = read_jsonl(windows_path)

    if args.prepare_only:
        return
    if not args.out_scored_jsonl or not args.out_summary_json or not args.out_md:
        raise SystemExit("scoring requires --out-scored-jsonl, --out-summary-json, and --out-md")
    scored_rows, summary = score_windows(
        rows,
        model_name=args.model_name,
        model_revision=args.model_revision,
        local_files_only=args.local_files_only,
        xcomet_code_dir=args.xcomet_code_dir,
        batch_size=args.batch_size,
        gpus=args.gpus,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        max_combined_tokens=args.max_combined_tokens,
    )
    write_jsonl(args.out_scored_jsonl, scored_rows)
    write_json(args.out_summary_json, summary)
    Path(args.out_md).write_text(markdown_summary(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

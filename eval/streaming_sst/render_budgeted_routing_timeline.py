"""Render a publication timeline from a budgeted AutoTerm evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "nlp": "#4F7CAC",
    "medicine": "#D9822B",
    "other": "#94A3B8",
    "covered": "#2F855A",
    "missing": "#E8A1A1",
}


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def domain_for_preset(preset: str) -> str:
    normalized = str(preset).strip().lower()
    if normalized.startswith("nlp_"):
        return "nlp"
    if normalized.startswith("medicine_"):
        return "medicine"
    return "other"


def talk_label(span: dict[str, Any]) -> str:
    item_id = str(span.get("item_id") or "")
    domain = str(span.get("expected_domain") or "").lower()
    suffix = item_id.rsplit(".", 1)[-1].replace("medicine_", "")
    prefix = "ACL" if domain == "nlp" else "Med" if domain == "medicine" else "Talk"
    return f"{prefix} {suffix}"


def run_length_intervals(
    records: Sequence[dict[str, Any]],
    *,
    value: Callable[[dict[str, Any]], Any],
    duration_s: float,
) -> list[tuple[float, float, Any]]:
    if not records:
        return []
    intervals: list[tuple[float, float, Any]] = []
    start_s = 0.0
    current = value(records[0])
    for record in records[1:]:
        next_value = value(record)
        if next_value == current:
            continue
        boundary_s = float(record.get("cursor_s") or start_s)
        intervals.append((start_s, boundary_s, current))
        start_s = boundary_s
        current = next_value
    intervals.append((start_s, duration_s, current))
    return intervals


def time_label(seconds: float) -> str:
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = int(seconds // 3600)
    minutes = int(round((seconds - hours * 3600) / 60))
    return f"{hours}h{minutes:02d}"


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def render(payload: dict[str, Any], output: Path, *, title: str) -> None:
    records = [dict(item) for item in payload.get("records") or []]
    spans = [dict(item) for item in payload.get("block_spans") or []]
    if not records or not spans:
        raise ValueError("input must contain non-empty records and block_spans")
    if not any(record.get("selected_slice_presets") for record in records):
        raise ValueError("input does not contain budgeted selected_slice_presets metadata")

    duration_s = float(payload.get("summary", {}).get("audio_seconds") or 0.0)
    if duration_s <= 0:
        duration_s = max(float(record.get("cursor_s") or 0.0) for record in records)

    img = Image.new("RGB", (1800, 720), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    title_font = load_font(40, bold=True)
    subtitle_font = load_font(22)
    label_font = load_font(22, bold=True)
    body_font = load_font(19)
    small_font = load_font(16)
    metric_font = load_font(25, bold=True)

    draw.text((70, 38), title, font=title_font, fill="#172033")
    draw.text(
        (72, 91),
        "The multi-slice working set can retain the correct glossary even while its pinned active slice adapts.",
        font=subtitle_font,
        fill="#556176",
    )

    left, right = 280, 1715
    scale = (right - left) / duration_s
    row_h = 58
    rows = {
        "true": 190,
        "top": 310,
        "covered": 430,
    }

    def x(seconds: float) -> int:
        return int(round(left + max(0.0, min(duration_s, seconds)) * scale))

    def fill_interval(start_s: float, end_s: float, y: int, color: str) -> None:
        x0, x1 = x(start_s), max(x(start_s) + 1, x(end_s))
        draw.rectangle((x0, y, x1, y + row_h), fill=color)

    draw.text((70, rows["true"] + 15), "True talk domain", font=label_font, fill="#334155")
    draw.text((70, rows["top"] + 15), "Pinned active slice", font=label_font, fill="#334155")
    draw.text((70, rows["covered"] + 4), "Correct slice in", font=label_font, fill="#334155")
    draw.text((70, rows["covered"] + 30), "top-4 working set", font=label_font, fill="#334155")

    for span in spans:
        start_s = float(span.get("start_sample") or 0) / 16000.0
        end_s = float(span.get("end_sample") or 0) / 16000.0
        domain = str(span.get("expected_domain") or "other").lower()
        fill_interval(start_s, end_s, rows["true"], COLORS.get(domain, COLORS["other"]))
        width = x(end_s) - x(start_s)
        label = talk_label(span)
        text_box = draw.textbbox((0, 0), label, font=small_font)
        text_width = text_box[2] - text_box[0]
        if width >= text_width + 12:
            draw.text(
                (x(start_s) + (width - text_width) / 2, rows["true"] + 19),
                label,
                font=small_font,
                fill="#FFFFFF",
            )

    top_intervals = run_length_intervals(
        records,
        value=lambda record: str(record.get("active_domain") or "other").lower(),
        duration_s=duration_s,
    )
    for start_s, end_s, domain in top_intervals:
        fill_interval(start_s, end_s, rows["top"], COLORS.get(str(domain), COLORS["other"]))

    def correct_slice_present(record: dict[str, Any]) -> bool:
        expected = str(record.get("expected_domain") or "").lower()
        selected = [domain_for_preset(item) for item in record.get("selected_slice_presets") or []]
        return expected in selected

    coverage_intervals = run_length_intervals(
        records,
        value=correct_slice_present,
        duration_s=duration_s,
    )
    for start_s, end_s, covered in coverage_intervals:
        fill_interval(
            start_s,
            end_s,
            rows["covered"],
            COLORS["covered"] if covered else COLORS["missing"],
        )

    for span in spans[1:]:
        boundary_s = float(span.get("start_sample") or 0) / 16000.0
        draw.line(
            (x(boundary_s), rows["true"] - 12, x(boundary_s), rows["covered"] + row_h + 14),
            fill="#D4DCE7",
            width=2,
        )

    tick_count = 6
    for idx in range(tick_count + 1):
        seconds = duration_s * idx / tick_count
        xpos = x(seconds)
        draw.line((xpos, rows["covered"] + row_h + 8, xpos, rows["covered"] + row_h + 16), fill="#64748B", width=2)
        label = time_label(seconds)
        box = draw.textbbox((0, 0), label, font=small_font)
        draw.text((xpos - (box[2] - box[0]) / 2, rows["covered"] + row_h + 23), label, font=small_font, fill="#64748B")

    legend_y = 138
    for idx, (label, color) in enumerate(
        [
            ("NLP", COLORS["nlp"]),
            ("Medicine", COLORS["medicine"]),
            ("Other pinned", COLORS["other"]),
            ("correct slice present", COLORS["covered"]),
        ]
    ):
        xpos = 950 + idx * 190
        draw.rounded_rectangle((xpos, legend_y, xpos + 22, legend_y + 22), radius=4, fill=color)
        draw.text((xpos + 31, legend_y - 1), label, font=small_font, fill="#475569")

    top1_accuracy = mean(
        1.0
        if str(record.get("active_domain") or "").lower()
        == str(record.get("expected_domain") or "").lower()
        else 0.0
        for record in records
    )
    coverage = mean(1.0 if correct_slice_present(record) else 0.0 for record in records)
    max_slices = max(int(record.get("selected_slice_count") or 0) for record in records)
    max_terms = max(int(record.get("selected_term_count") or 0) for record in records)
    admission_delays: list[float] = []
    for span in spans[1:]:
        boundary_s = float(span.get("start_sample") or 0) / 16000.0
        admitted = next(
            (
                record
                for record in records
                if float(record.get("cursor_s") or 0.0) >= boundary_s
                and correct_slice_present(record)
            ),
            None,
        )
        if admitted is not None:
            admission_delays.append(
                max(0.0, float(admitted.get("cursor_s") or 0.0) - boundary_s)
            )
    admission_value = (
        f"{max(admission_delays):.1f}s max" if admission_delays else "n/a"
    )

    metrics = [
        ("session", f"{len(spans)} talks / {duration_s / 3600:.2f}h"),
        ("active budget", f"{max_slices} slices / {max_terms:,} terms"),
        ("active-domain agreement", f"{100 * top1_accuracy:.1f}%"),
        ("correct-slice coverage", f"{100 * coverage:.1f}%"),
        ("correct-slice admission", admission_value),
    ]
    card_y, card_h = 590, 92
    card_gap = 18
    card_w = int((right - left - card_gap * (len(metrics) - 1)) / len(metrics))
    for idx, (name, value) in enumerate(metrics):
        x0 = left + idx * (card_w + card_gap)
        draw.rounded_rectangle(
            (x0, card_y, x0 + card_w, card_y + card_h),
            radius=12,
            fill="#F8FAFC",
            outline="#CBD5E1",
            width=2,
        )
        draw.text((x0 + 15, card_y + 12), name, font=small_font, fill="#64748B")
        draw.text((x0 + 15, card_y + 43), value, font=metric_font, fill="#172033")

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--title",
        default="Budgeted AutoTerm routing across a ten-talk stream",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    render(payload, Path(args.output), title=args.title)


if __name__ == "__main__":
    main()

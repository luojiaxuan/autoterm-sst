"""Render a publication timeline from a budgeted AutoTerm evaluation JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

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


def render(payload: dict[str, Any], output: Path) -> None:
    records = [dict(item) for item in payload.get("records") or []]
    spans = [dict(item) for item in payload.get("block_spans") or []]
    if not records or not spans:
        raise ValueError("input must contain non-empty records and block_spans")
    if not any(record.get("selected_slice_presets") for record in records):
        raise ValueError("input does not contain budgeted selected_slice_presets metadata")

    duration_s = float(payload.get("summary", {}).get("audio_seconds") or 0.0)
    if duration_s <= 0:
        duration_s = max(float(record.get("cursor_s") or 0.0) for record in records)

    img = Image.new("RGB", (1800, 470), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    label_font = load_font(22, bold=True)
    cell_font = load_font(17, bold=True)
    small_font = load_font(16)

    left, right = 330, 1730
    talk_width = (right - left) / len(spans)
    row_h = 58
    rows = {
        "true": 70,
        "top": 190,
        "covered": 310,
    }

    def x(seconds: float) -> int:
        clipped = max(0.0, min(duration_s, seconds))
        for talk_index, span in enumerate(spans):
            start_s = float(span.get("start_sample") or 0) / 16000.0
            end_s = float(span.get("end_sample") or 0) / 16000.0
            if clipped <= end_s or talk_index == len(spans) - 1:
                fraction = 0.0 if end_s <= start_s else (clipped - start_s) / (end_s - start_s)
                fraction = max(0.0, min(1.0, fraction))
                return int(round(left + (talk_index + fraction) * talk_width))
        return right

    def fill_interval(start_s: float, end_s: float, y: int, color: str) -> None:
        x0, x1 = x(start_s), max(x(start_s) + 1, x(end_s))
        draw.rectangle((x0, y, x1, y + row_h), fill=color)

    draw.text((70, rows["true"] + 15), "Talk domain", font=label_font, fill="#334155")
    draw.text((70, rows["top"] + 15), "Selected slice", font=label_font, fill="#334155")
    draw.text((70, rows["covered"] + 4), "Correct slice in", font=label_font, fill="#334155")
    draw.text((70, rows["covered"] + 30), "working set", font=label_font, fill="#334155")

    for span in spans:
        start_s = float(span.get("start_sample") or 0) / 16000.0
        end_s = float(span.get("end_sample") or 0) / 16000.0
        domain = str(span.get("expected_domain") or "other").lower()
        fill_interval(start_s, end_s, rows["true"], COLORS.get(domain, COLORS["other"]))
        width = x(end_s) - x(start_s)
        label = "NLP" if domain == "nlp" else "Medicine" if domain == "medicine" else "Other"
        text_box = draw.textbbox((0, 0), label, font=cell_font)
        text_width = text_box[2] - text_box[0]
        if width >= text_width + 12:
            draw.text(
                (x(start_s) + (width - text_width) / 2, rows["true"] + 18),
                label,
                font=cell_font,
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

    for talk_index, span in enumerate(spans):
        start_s = float(span.get("start_sample") or 0) / 16000.0
        end_s = float(span.get("end_sample") or 0) / 16000.0
        center = (x(start_s) + x(end_s)) / 2
        talk_text = f"Talk {talk_index + 1}"
        box = draw.textbbox((0, 0), talk_text, font=small_font)
        draw.text(
            (center - (box[2] - box[0]) / 2, rows["covered"] + row_h + 17),
            talk_text,
            font=small_font,
            fill="#64748B",
        )

    for span in spans[1:]:
        boundary_s = float(span.get("start_sample") or 0) / 16000.0
        draw.line(
            (x(boundary_s), rows["true"] - 8, x(boundary_s), rows["covered"] + row_h + 8),
            fill="#D4DCE7",
            width=2,
        )

    for y in rows.values():
        draw.rectangle((left, y, right, y + row_h), outline="#D4DCE7", width=2)

    legend_y = 20
    for idx, (label, color) in enumerate(
        [
            ("NLP", COLORS["nlp"]),
            ("Medicine", COLORS["medicine"]),
            ("Other", COLORS["other"]),
            ("Correct slice active", COLORS["covered"]),
            ("Absent", COLORS["missing"]),
        ]
    ):
        xpos = 600 + idx * 220
        draw.rounded_rectangle((xpos, legend_y, xpos + 22, legend_y + 22), radius=4, fill=color)
        draw.text((xpos + 31, legend_y - 1), label, font=small_font, fill="#475569")

    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    render(payload, Path(args.output))


if __name__ == "__main__":
    main()

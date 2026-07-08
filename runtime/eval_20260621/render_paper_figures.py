from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "demo_paper_emnlp" / "latex" / "figures"
UI_SOURCE = ROOT / "runtime" / "eval_20260621" / "figure2_ui.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE = font(42, True)
SUBTITLE = font(24)
HEAD = font(26, True)
BODY = font(22)
SMALL = font(18)
TINY = font(15)


def text_size(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=face)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.split("\n"):
        words = raw_line.split()
        line = ""
        for word in words:
            trial = word if not line else f"{line} {word}"
            if text_size(draw, trial, face)[0] <= width:
                line = trial
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
    return lines


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, face: ImageFont.ImageFont, fill: str) -> None:
    draw.text(xy, text, font=face, fill=fill)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    body: list[str],
    fill: str,
    outline: str,
    title_fill: str = "#1F2937",
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=24, fill=fill, outline=outline, width=3)
    draw.text((x0 + 26, y0 + 22), title, font=HEAD, fill=title_fill)
    y = y0 + 72
    for item in body:
        for line_idx, line in enumerate(wrap_text(draw, item, BODY, x1 - x0 - 72)):
            if line_idx == 0:
                draw.ellipse((x0 + 30, y + 8, x0 + 40, y + 18), fill=outline)
                text_x = x0 + 52
            else:
                text_x = x0 + 52
            draw.text((text_x, y), line, font=BODY, fill="#374151")
            y += 32
        y += 6


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#475569",
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length = 18
    spread = 0.45
    left = (end[0] - length * math.cos(angle - spread), end[1] - length * math.sin(angle - spread))
    right = (end[0] - length * math.cos(angle + spread), end[1] - length * math.sin(angle + spread))
    draw.polygon([end, left, right], fill=color)


def render_architecture() -> None:
    img = Image.new("RGB", (1800, 1050), "#F8FAFC")
    draw = ImageDraw.Draw(img)
    draw.text((70, 48), "AutoTerm-SST Runtime Architecture", font=TITLE, fill="#0F172A")
    draw.text(
        (72, 102),
        "A thin streaming framework delegates terminology retrieval, routing, prompting, and model serving to swappable agents.",
        font=SUBTITLE,
        fill="#475569",
    )

    rounded_box(
        draw,
        (70, 180, 410, 590),
        "Inputs and UI",
        [
            "Web or Electron client",
            "Microphone, file, or system audio",
            "Manual terms remain optional",
            "Mock mode uses the same protocol",
        ],
        "#E0F2FE",
        "#0284C7",
    )
    rounded_box(
        draw,
        (470, 180, 825, 590),
        "Thin framework",
        [
            "Static UI and REST controls",
            "WebSocket audio stream",
            "Session lifecycle tracking",
            "agent_type routes to an agent",
        ],
        "#DCFCE7",
        "#16A34A",
    )
    rounded_box(
        draw,
        (885, 180, 1315, 590),
        "Agent boundary",
        [
            "Audio buffer and micro-batcher",
            "MaxSim retrieval plugin",
            "Hybrid active-slice router",
            "PromptBuilder with top-10 terms",
            "Qwen3-Omni served through vLLM",
        ],
        "#F3E8FF",
        "#7C3AED",
    )
    rounded_box(
        draw,
        (1375, 180, 1730, 590),
        "Terminology resources",
        [
            "Manifest-driven snapshots",
            "Domain slices: NLP, medicine, finance, legal",
            "Broad open memories as fallback pools",
            "Curated ACL glossary for diagnostics",
        ],
        "#FEF3C7",
        "#D97706",
    )

    rounded_box(
        draw,
        (250, 730, 1560, 950),
        "JSON evidence stream",
        [
            "translation text",
            "retrieved term pairs and scores",
            "active slice, router action, confidence, switch count",
            "retrieval and generation latency",
        ],
        "#FFFFFF",
        "#64748B",
    )

    arrow(draw, (410, 385), (470, 385))
    arrow(draw, (825, 385), (885, 385))
    arrow(draw, (1375, 385), (1315, 385))
    arrow(draw, (1100, 590), (1100, 730))
    arrow(draw, (520, 730), (260, 590))

    draw.rounded_rectangle((79, 604, 1730, 675), radius=18, fill="#EEF2FF", outline="#6366F1", width=2)
    draw.text((112, 625), "Control plane: start/stop, latency mode, terminology mode, manual presets, health checks", font=BODY, fill="#334155")
    arrow(draw, (640, 590), (640, 604), "#6366F1", 3)
    arrow(draw, (1110, 604), (1110, 590), "#6366F1", 3)

    img.save(FIG_DIR / "autoterm_architecture.png")


def render_routing_timeline() -> None:
    img = Image.new("RGB", (1600, 660), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((70, 42), "4-block real E2E routing probe", font=TITLE, fill="#0F172A")
    draw.text(
        (72, 95),
        "Preliminary system evidence: active glossary switches follow ACL/medicine domain transitions without wrong switches.",
        font=SUBTITLE,
        fill="#475569",
    )

    left, right = 220, 1490
    scale = (right - left) / 480.0
    y_true, y_active = 205, 335
    bar_h = 58
    colors = {"nlp": "#4F7CAC", "medicine": "#D88439"}
    light = {"nlp": "#D8E6F5", "medicine": "#F8DFC5"}

    def x(t: float) -> int:
        return int(left + t * scale)

    def segment(y: int, start: float, end: float, domain: str, label: str, pale: bool = False) -> None:
        fill = light[domain] if pale else colors[domain]
        draw.rounded_rectangle((x(start), y, x(end), y + bar_h), radius=12, fill=fill, outline="#FFFFFF", width=3)
        tw, th = text_size(draw, label, BODY)
        draw.text((x(start) + (x(end) - x(start) - tw) / 2, y + (bar_h - th) / 2 - 2), label, font=BODY, fill="#FFFFFF" if not pale else "#334155")

    draw.text((70, y_true + 15), "True domain", font=BODY, fill="#334155")
    draw.text((70, y_active + 15), "Active slice", font=BODY, fill="#334155")

    segment(y_true, 0, 120, "nlp", "ACL / NLP")
    segment(y_true, 120, 240, "medicine", "Medicine")
    segment(y_true, 240, 360, "nlp", "ACL / NLP")
    segment(y_true, 360, 480, "medicine", "Medicine")

    segment(y_active, 0, 140.16, "nlp", "nlp_core_10k")
    segment(y_active, 140.16, 257.28, "medicine", "medicine_core_10k")
    segment(y_active, 257.28, 397.44, "nlp", "nlp_core_10k")
    segment(y_active, 397.44, 480, "medicine", "medicine_core_10k")

    for t in [0, 120, 240, 360, 480]:
        draw.line((x(t), y_true - 28, x(t), y_active + bar_h + 35), fill="#CBD5E1", width=2)
        label = f"{int(t)}s"
        tw, _ = text_size(draw, label, SMALL)
        draw.text((x(t) - tw / 2, y_active + bar_h + 45), label, font=SMALL, fill="#64748B")

    switches = [(140.16, "+20.16s"), (257.28, "+17.28s"), (397.44, "+37.44s")]
    for t, label in switches:
        draw.line((x(t), y_true - 16, x(t), y_active + bar_h + 18), fill="#111827", width=3)
        draw.line((x(t) - 7, y_true - 16, x(t) + 7, y_true - 16), fill="#111827", width=3)
        tw, _ = text_size(draw, label, SMALL)
        draw.rounded_rectangle((x(t) - tw / 2 - 12, y_true - 62, x(t) + tw / 2 + 12, y_true - 28), radius=10, fill="#F8FAFC", outline="#94A3B8")
        draw.text((x(t) - tw / 2, y_true - 57), label, font=SMALL, fill="#111827")

    metrics = [
        ("transitions", "3"),
        ("switches", "3"),
        ("wrong switches", "0"),
        ("steady-state acc.", "0.9947"),
        ("retrieval p95", "88.29 ms"),
    ]
    x0 = 130
    for name, value in metrics:
        w = 235 if name != "steady-state acc." else 275
        draw.rounded_rectangle((x0, 515, x0 + w, 595), radius=18, fill="#F8FAFC", outline="#CBD5E1", width=2)
        draw.text((x0 + 18, 530), name, font=TINY, fill="#64748B")
        draw.text((x0 + 18, 555), value, font=HEAD, fill="#0F172A")
        x0 += w + 22

    img.save(FIG_DIR / "routing_timeline_4block.png")


def render_ui_crop() -> None:
    img = Image.open(UI_SOURCE).convert("RGB")
    crop = img.crop((70, 185, 1300, 900))
    crop.save(FIG_DIR / "ui_evidence_panel.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    render_architecture()
    render_routing_timeline()
    render_ui_crop()


if __name__ == "__main__":
    main()

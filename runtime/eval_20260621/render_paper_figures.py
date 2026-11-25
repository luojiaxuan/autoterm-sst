from __future__ import annotations

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


def render_routing_timeline() -> None:
    img = Image.new("RGB", (1600, 660), "#FFFFFF")
    draw = ImageDraw.Draw(img)
    draw.text((70, 42), "4-block real E2E routing probe", font=TITLE, fill="#0F172A")
    draw.text(
        (72, 95),
        "Active glossary switches follow ACL/medicine domain transitions without wrong switches.",
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
        ("retrieval p95", "101.64 ms"),
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
    # note (luojiaxuan): Figure 1 is maintained as editable SVG and exported to PDF/PNG from that single vector source.
    render_routing_timeline()
    render_ui_crop()


if __name__ == "__main__":
    main()

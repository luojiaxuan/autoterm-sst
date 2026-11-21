"""Generate the AutoTerm-SST home screen icons with Pillow."""

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "serve" / "static" / "icons"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (1024, 1024), "#1a1a2e")
    draw = ImageDraw.Draw(image)
    # Coordinates are doubled for antialiasing. Both bubbles fit the maskable safe zone.
    draw.rounded_rectangle((288, 296, 606, 500), radius=48, fill="#4facfe")
    draw.polygon([(326, 476), (326, 548), (408, 476)], fill="#4facfe")
    draw.rounded_rectangle((418, 444, 736, 648), radius=48, fill="#00f2fe")
    draw.polygon([(616, 622), (698, 698), (698, 622)], fill="#00f2fe")
    draw.rounded_rectangle((478, 508, 676, 528), radius=10, fill="#1a1a2e")
    draw.rounded_rectangle((478, 564, 618, 584), radius=10, fill="#1a1a2e")

    for size in (180, 192, 512):
        image.resize((size, size), Image.Resampling.LANCZOS).save(out_dir / f"autoterm-{size}.png")


if __name__ == "__main__":
    main()

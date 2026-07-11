#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
DATA = HERE / "data.tsv"
OUT = HERE.parents[1] / "latex" / "figures" / "rag_compute_rtf_compact.pdf"


def main() -> None:
    with DATA.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    lm = [int(row["lm"]) for row in rows]
    mean_rtf = [float(row["rag_mean_rtf_pct"]) for row in rows]
    median_ms = [float(row["rag_median_ms"]) for row in rows]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.48), constrained_layout=True)
    panels = (
        (axes[0], mean_rtf, "Mean RTF (\\%)", "#D6604D", "o", (0.8, 6.1)),
        (axes[1], median_ms, "Median call (ms)", "#3288BD", "D", (35.0, 45.0)),
    )
    for panel_index, (axis, values, ylabel, color, marker, ylim) in enumerate(panels):
        axis.plot(lm, values, color=color, marker=marker, linewidth=1.6, markersize=4.2)
        axis.set_xticks(lm)
        axis.set_xlabel("Latency multiplier (LM)")
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.grid(True, linestyle=":", linewidth=0.55, alpha=0.55)
        axis.text(
            0.03,
            0.95,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()

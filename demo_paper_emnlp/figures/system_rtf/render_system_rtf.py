#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent
SYSTEM_DATA = HERE / "system_rtf.tsv"
LM_DATA = HERE / "lm_system_rtf.tsv"
OUT = HERE.parents[1] / "latex" / "figures" / "system_rtf_scaling_compact.pdf"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    system_rows = read_tsv(SYSTEM_DATA)
    lm_rows = read_tsv(LM_DATA)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.52), constrained_layout=True)

    lm = [int(row["lm"]) for row in lm_rows]
    maxsim = [float(row["maxsim_component_ratio"]) for row in lm_rows]
    other = [float(row["system_rtf"]) - value for row, value in zip(lm_rows, maxsim)]
    axes[0].bar(lm, other, bottom=maxsim, color="#A7A9AC", width=0.65, label="Other system")
    axes[0].bar(lm, maxsim, color="#D6604D", width=0.65, label="MaxSim")
    axes[0].set_xticks(lm)
    axes[0].set_xlabel("Latency multiplier (LM)")
    axes[0].set_ylabel("System RTF")
    axes[0].set_ylim(0.0, 0.48)
    axes[0].legend(loc="upper right", fontsize=5.8, frameon=False, handlelength=1.0)
    for x_value, row in zip(lm, lm_rows):
        total = float(row["system_rtf"])
        axes[0].text(x_value, total + 0.010, f"{total:.3f}", ha="center", fontsize=6.0)

    label_map = {
        "Known": "Known",
        "AutoTerm": "Auto-\nTerm",
        "Merged-100k": "100k",
        "Merged-1M": "1M",
    }
    labels = [label_map[row["label"]] for row in system_rows]
    rtfs = [float(row["system_rtf"]) for row in system_rows]
    colors = ["#7B9EBD", "#D6604D", "#A7A9AC", "#6F7175"]
    bars = axes[1].bar(range(len(rtfs)), rtfs, color=colors, width=0.68)
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_xlabel("Ten-talk policy (LM=2)")
    axes[1].set_ylabel("System RTF")
    axes[1].set_ylim(0.0, 0.48)
    for bar, value in zip(bars, rtfs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.012,
            f"{value:.3f}",
            ha="center",
            va="top",
            fontsize=6.2,
        )

    for panel_index, axis in enumerate(axes):
        axis.grid(axis="y", linestyle=":", linewidth=0.55, alpha=0.55)
        axis.text(
            0.03,
            0.96,
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

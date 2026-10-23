# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from khisto.matplotlib import hist

SEED = 42
SAMPLE_SIZE = 10_000


def save_binning_comparison_figure(output_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    # The reciprocal square of a standard normal follows a Lévy distribution.
    levy = 1 / rng.standard_normal(SAMPLE_SIZE) ** 2
    ages = np.loadtxt(
        Path(__file__).resolve().parents[1]
        / "docs" / "datasets" / "california-housing-ages.csv",
        dtype=int,
    )
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 12,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    }):
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 5.8), sharex="row", sharey="row", facecolor="white"
        )
        fig.subplots_adjust(
            left=0.10, right=0.98, bottom=0.10, top=0.80, hspace=0.72, wspace=0.18
        )
        for column, (title, subtitle, color, text_color) in enumerate(
            [
                ("Before", "Equal-width bins", "black", "white"),
                ("After", "Khisto", "#ff7900", "black"),
            ]
        ):
            position = axes[0, column].get_position()
            center = (position.x0 + position.x1) / 2
            fig.add_artist(Rectangle(
                (position.x0, 0.915), position.width, 0.075,
                transform=fig.transFigure, facecolor=color, edgecolor="none",
            ))
            fig.text(
                center, 0.9525, title, ha="center", va="center",
                fontsize=22, weight="bold", color=text_color,
            )
            fig.text(center, 0.875, subtitle, ha="center", fontsize=13)

        for row, (data, bins, label, xlabel) in enumerate(
            [
                (levy, 100, "Lévy", "Value"),
                (ages, 35, "California Housing", "Median house age (years)"),
            ]
        ):
            axes[row, 0].hist(
                data, bins=bins, density=True, color="#666666",
                edgecolor="white", linewidth=0.6,
            )
            hist(data, density=True, ax=axes[row, 1], color="#f16e00")
            position = axes[row, 0].get_position()
            fig.text(
                position.x0, position.y1 + 0.025, label, fontsize=13, weight="bold",
            )
            fig.text(
                0.54, position.y0 - 0.078, xlabel, ha="center", fontsize=12,
            )
            axes[row, 0].set_ylabel("Density")
            for ax in axes[row]:
                ax.set_facecolor("white")
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["bottom", "left"]].set_color("#666666")
                ax.tick_params(labelsize=11, color="#666666", which="both")
                ax.minorticks_off()

        for ax in axes[0]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.minorticks_off()
        axes[1, 0].set_ylim(0, 0.07)
        spike = axes[1, 0].patches[10]
        axes[1, 0].annotate(
            "Two ages in one bin\ncreate an artificial spike",
            xy=(spike.get_x() + spike.get_width() / 2, spike.get_height()),
            xytext=(0.65, 0.93), textcoords="axes fraction",
            ha="center", va="top", fontsize=10,
            arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 0.8},
        )
        fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)


def save_counts_density_comparison_figure(output_path: Path) -> None:
    equal_edges = np.array([0, 10, 20, 30, 40])
    equal_counts = np.array([90, 110, 100, 200])
    variable_edges = np.array([0, 30, 40])
    variable_counts = np.array([300, 200])
    total_count = equal_counts.sum()

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True)
    configurations = (
        (equal_edges, equal_counts, "Without Khisto\nEqual-width bins"),
        (variable_edges, variable_counts, "With Khisto\nVariable-width bins"),
    )
    for column, (edges, counts, title) in enumerate(configurations):
        widths = np.diff(edges)
        densities = counts / (total_count * widths)
        colors = ["darkorange" if edge >= 30 else "steelblue" for edge in edges[:-1]]
        count_bars = axes[0, column].bar(
            edges[:-1], counts, width=widths, align="edge", color=colors,
        )
        density_bars = axes[1, column].bar(
            edges[:-1], densities, width=widths, align="edge", color=colors,
        )
        axes[0, column].bar_label(count_bars, labels=[str(count) for count in counts], padding=3)
        axes[1, column].bar_label(
            density_bars,
            labels=[f"{density:.3f}" for density in densities],
            padding=3,
        )
        axes[0, column].set_title(title)
        for row in range(2):
            axes[row, column].set_xticks([0, 10, 20, 30, 40])
            axes[row, column].tick_params(axis="x", labelbottom=True)
            axes[row, column].set_xlabel("Value")
        axes[0, column].set_ylim(0, 350)
        axes[1, column].set_ylim(0, 0.048)

    axes[0, 0].set_ylabel("Count")
    axes[1, 0].set_ylabel("Density")
    axes[0, 0].text(
        15,
        165,
        "Small local fluctuation",
        ha="center",
    )
    axes[0, 1].text(
        15,
        245,
        "Counts apply to the whole interval;\ndo not rely on rectangle area",
        color="white",
        ha="center",
        va="center",
        weight="bold",
        fontsize=9,
    )
    axes[1, 1].text(
        15,
        0.012,
        "Fluctuation smoothed out\nin the wider bin",
        color="white",
        ha="center",
        va="center",
        weight="bold",
    )
    axes[1, 1].annotate(
        "bin 30–40 is twice \nas dense as bin 0-30",
        xy=(35, 0.034),
        xytext=(15, 0.034),
        arrowprops={"arrowstyle": "->", "color": "#333333"},
        ha="center",
        va="center",
        weight="bold",
    )
    fig.suptitle("Same 500 observations, two binning choices")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    images_dir = repo_root / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    save_binning_comparison_figure(images_dir / "binning-comparison.png")
    save_counts_density_comparison_figure(images_dir / "counts-vs-density.png")

    print(f"Wrote {images_dir / 'binning-comparison.png'}")
    print(f"Wrote {images_dir / 'counts-vs-density.png'}")


if __name__ == "__main__":
    main()

# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

from khisto.matplotlib import hist

SEED = 42
SAMPLE_SIZE = 10_000
BASELINE_COLOR = "#9373bd"


def save_binning_comparison_figure(output_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    # The reciprocal square of a standard normal follows a Lévy distribution.
    levy = 1 / rng.standard_normal(SAMPLE_SIZE) ** 2
    ages = np.loadtxt(
        Path(__file__).resolve().parents[1]
        / "docs" / "datasets" / "adult-ages.csv",
        dtype=int,
    )
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "svg.fonttype": "path",
        "svg.hashsalt": "khisto-comparison",
    }):
        fig, axes = plt.subplots(
            2, 2, figsize=(10, 6), sharex="row", sharey="row", facecolor="white"
        )
        fig.subplots_adjust(
            left=0.105, right=0.965, bottom=0.115, top=0.81,
            hspace=0.80, wspace=0.32,
        )
        for column, title in enumerate(("Before / Equal-width", "After / Khisto")):
            fig.text(
                axes[0, column].get_position().x0, 0.95, title,
                fontsize=16, weight="bold", va="top",
            )
        for row, (data, label, xlabel) in enumerate(
            [
                (levy, "Lévy distribution", "Value"),
                (ages, "UCI Adult", "Age (years)"),
            ]
        ):
            _, adaptive_edges, _ = hist(
                data, density=True, ax=axes[row, 1], color="#f16e00"
            )
            bin_count = len(adaptive_edges) - 1
            axes[row, 0].hist(
                data, bins=bin_count, density=True, color=BASELINE_COLOR,
                edgecolor="white", linewidth=0.6,
            )
            position = axes[row, 0].get_position()
            fig.text(
                position.x0, position.y1 + 0.035, label,
                fontsize=13, weight="bold", color="#333333", va="bottom",
            )
            for ax in axes[row]:
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Density")
                ax.set_facecolor("none")
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["bottom", "left"]].set_color("#666666")
                ax.tick_params(
                    labelsize=10, color="#666666", which="both", labelleft=True
                )
                ax.minorticks_off()

        for ax in axes[0]:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.minorticks_off()
        fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
        for text in fig.texts:
            text.set_visible(False)
        for theme, foreground, axis_color, orange in (
            ("light", "#000000", "#666666", "#f16e00"),
            ("dark", "#ffffff", "#999999", "#ff7900"),
        ):
            for row, dataset in enumerate(("levy", "adult")):
                for column, method in enumerate(("equal-width", "khisto")):
                    ax = axes[row, column]
                    ax.xaxis.label.set_color(foreground)
                    ax.yaxis.label.set_color(foreground)
                    ax.tick_params(colors=foreground, which="both")
                    ax.spines[["bottom", "left"]].set_color(axis_color)
                    for bar in ax.patches:
                        bar.set_facecolor((BASELINE_COLOR, orange)[column])
                        bar.set_edgecolor((BASELINE_COLOR, orange)[column])
                    position = ax.get_position()
                    bounds = Bbox.from_extents(
                        position.x0 - 0.085, position.y0 - 0.085,
                        position.x1 + 0.015, position.y1 + 0.015,
                    ).transformed(fig.transFigure - fig.dpi_scale_trans)
                    svg_path = output_path.parent / f"{dataset}-{method}-{theme}.svg"
                    fig.savefig(
                        svg_path,
                        bbox_inches=bounds, transparent=True,
                        metadata={"Date": None},
                    )
                    svg_path.write_text(
                        "\n".join(
                            line.rstrip()
                            for line in svg_path.read_text(encoding="utf-8").splitlines()
                        ) + "\n",
                        encoding="utf-8",
                    )
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

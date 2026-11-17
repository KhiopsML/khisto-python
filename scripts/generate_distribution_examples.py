# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    for suffix, foreground, secondary, axis_color, orange in (
        ("", "black", "#646776", "#666666", "#f16e00"),
        ("-dark", "white", "#9ca0a5", "#999999", "#ff7900"),
    ):
        with plt.rc_context({
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "mathtext.default": "regular",
            "svg.fonttype": "path",
            "svg.hashsalt": "khisto-comparison",
            "text.color": foreground,
            "axes.labelcolor": foreground,
            "xtick.color": foreground,
            "ytick.color": foreground,
        }):
            fig, axes = plt.subplots(
                2, 2, figsize=(10, 6.59), sharex="row", sharey="row",
            )
            fig.subplots_adjust(
                left=0.087, right=0.985, bottom=0.08, top=0.742,
                hspace=0.85, wspace=0.355,
            )
            for column, (title, subtitle, color) in enumerate(
                [("Before", "Equal-width", BASELINE_COLOR), ("After", "Khisto", "#ff7900")]
            ):
                left = column * 0.516
                fig.add_artist(plt.Line2D(
                    [left, left + 0.484], [0.996, 0.996],
                    transform=fig.transFigure, color=color, linewidth=3,
                ))
                fig.text(left, 0.973, title, fontsize=24, weight="bold", va="top")
                fig.text(left, 0.905, subtitle, fontsize=15, color=secondary, va="top")
            for row, (data, label, xlabel) in enumerate(
                [
                    (levy, "Lévy distribution", "Value"),
                    (ages, "Adult age", "Age (years)"),
                ]
            ):
                _, adaptive_edges, _ = hist(
                    data, density=True, ax=axes[row, 1],
                    color=orange, edgecolor=orange,
                )
                axes[row, 0].hist(
                    data, bins=len(adaptive_edges) - 1, density=True,
                    color=BASELINE_COLOR, edgecolor=BASELINE_COLOR, linewidth=0.6,
                )
                fig.text(
                    0, (0.819, 0.389)[row], label,
                    fontsize=18, weight="bold", va="top",
                )
                for ax in axes[row]:
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Density")
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.spines[["bottom", "left"]].set_color(axis_color)
                    ax.tick_params(
                        labelsize=10, color=axis_color, which="both", labelleft=True,
                    )
                    ax.minorticks_off()
                    if row == 1:
                        ax.locator_params(axis="x", nbins=5)
            for ax in axes[0]:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.minorticks_off()
            fig.savefig(
                output_path.with_stem(f"{output_path.stem}{suffix}"),
                transparent=True, metadata={"Date": None},
            )
            if not suffix:
                fig.savefig(
                    output_path.with_suffix(".png"),
                    dpi=180, facecolor="white", transparent=False,
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

    save_binning_comparison_figure(images_dir / "binning-comparison.svg")
    save_counts_density_comparison_figure(images_dir / "counts-vs-density.png")

    print(f"Wrote {images_dir / 'binning-comparison.svg'}")
    print(f"Wrote {images_dir / 'binning-comparison-dark.svg'}")
    print(f"Wrote {images_dir / 'binning-comparison.png'}")
    print(f"Wrote {images_dir / 'counts-vs-density.png'}")


if __name__ == "__main__":
    main()

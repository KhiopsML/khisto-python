# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from khisto.matplotlib import hist

SEED = 42
SAMPLE_SIZE = 10_000
PARETO_SHAPE = 3.0


def generate_gaussian_data(size: int = SAMPLE_SIZE, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=size)


def generate_pareto_data(
    size: int = SAMPLE_SIZE,
    shape: float = PARETO_SHAPE,
    seed: int = SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.pareto(a=shape, size=size) + 1.0


def save_gaussian_figure(data: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hist(data, density=True, ax=ax, color="steelblue", edgecolor="white", linewidth=0.8)
    ax.set_title("Adaptive histogram on a standard Gaussian")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
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
            edgecolor="white",
        )
        density_bars = axes[1, column].bar(
            edges[:-1], densities, width=widths, align="edge", color=colors,
            edgecolor="white",
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
    left_column_right = axes[0, 0].get_position().x1
    right_column_left = axes[0, 1].get_position().x0
    separator_x = (left_column_right + right_column_left) / 2
    fig.add_artist(Line2D(
        [separator_x, separator_x],
        [axes[1, 0].get_position().y0, axes[0, 0].get_position().y1],
        transform=fig.transFigure,
        color="#777777",
        linewidth=1,
        linestyle="--",
    ))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_pareto_figure(data: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    hist(data, density=True, ax=ax, color="darkorange", edgecolor="white", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Adaptive histogram on a heavy-tailed Pareto law")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    images_dir = repo_root / "docs" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    gaussian_data = generate_gaussian_data()
    pareto_data = generate_pareto_data()

    save_gaussian_figure(gaussian_data, images_dir / "gaussian-quick-start.png")
    save_counts_density_comparison_figure(
        images_dir / "counts-vs-density.png"
    )
    save_pareto_figure(pareto_data, images_dir / "pareto-quick-start.png")

    print(f"Gaussian sample: size={gaussian_data.size}, mean={gaussian_data.mean():.4f}, std={gaussian_data.std():.4f}")
    print(
        "Pareto sample: "
        f"size={pareto_data.size}, min={pareto_data.min():.4f}, max={pareto_data.max():.4f}, shape={PARETO_SHAPE:.1f}"
    )
    print(f"Wrote {images_dir / 'gaussian-quick-start.png'}")
    print(f"Wrote {images_dir / 'counts-vs-density.png'}")
    print(f"Wrote {images_dir / 'pareto-quick-start.png'}")


if __name__ == "__main__":
    main()

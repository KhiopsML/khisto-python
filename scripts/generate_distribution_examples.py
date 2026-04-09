# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    save_pareto_figure(pareto_data, images_dir / "pareto-quick-start.png")

    print(f"Gaussian sample: size={gaussian_data.size}, mean={gaussian_data.mean():.4f}, std={gaussian_data.std():.4f}")
    print(
        "Pareto sample: "
        f"size={pareto_data.size}, min={pareto_data.min():.4f}, max={pareto_data.max():.4f}, shape={PARETO_SHAPE:.1f}"
    )
    print(f"Wrote {images_dir / 'gaussian-quick-start.png'}")
    print(f"Wrote {images_dir / 'pareto-quick-start.png'}")


if __name__ == "__main__":
    main()

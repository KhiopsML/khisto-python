# Khisto

[![CI](https://github.com/KhiopsML/khisto-python/actions/workflows/ci.yaml/badge.svg)](https://github.com/KhiopsML/khisto-python/actions/workflows/ci.yaml)
[![Docs](https://github.com/KhiopsML/khisto-python/actions/workflows/docs.yaml/badge.svg)](https://khiopsml.github.io/khisto-python/)
[![Release](https://img.shields.io/github/v/release/KhiopsML/khisto-python?include_prereleases)](https://github.com/KhiopsML/khisto-python/releases)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://github.com/KhiopsML/khisto-python/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause--Clear-blue)](LICENSE)

**Optimal Binning Histograms for Python**

Khisto is a Python library for creating histograms using the **Khiops optimal binning algorithm**. Unlike standard histograms that use fixed-width bins or simple heuristics, Khisto automatically determines the optimal number of bins and their variable widths to best represent the underlying data distribution.

Documentation is available at **[khiopsml.github.io/khisto-python](https://khiopsml.github.io/khisto-python/)**.

| Standard Gaussian | Heavy-tailed Pareto |
| --- | --- |
| ![Adaptive Gaussian histogram](https://raw.githubusercontent.com/KhiopsML/khisto-python/main/docs/images/gaussian-quick-start.png) | ![Adaptive Pareto histogram](https://raw.githubusercontent.com/KhiopsML/khisto-python/main/docs/images/pareto-quick-start.png) |

## Installation

```bash
pip install khisto
```

With matplotlib support:

```bash
pip install "khisto[matplotlib]"
```

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

# Generate 10,000 samples from a Normal distribution
normal_data = np.random.normal(size=10000)

# Plot an adaptive histogram
n, bins, patches = hist(normal_data)
plt.show()

# Generate 10,000 samples from a Pareto distribution
long_tail_data = np.random.pareto(3, size=10000)

# Plot an adaptive histogram on logarithmic axes.
n, bins, patches = hist(long_tail_data)
plt.xscale("symlog")
plt.yscale("log")
plt.show()
```

## Development

```bash
# Clone repository
git clone https://github.com/KhiopsML/khisto-python.git
cd khisto-python

# Install with dev dependencies
uv sync --group dev --extra all

# Run tests
uv run pytest
```

## License

[BSD 3-Clause Clear License](LICENSE)

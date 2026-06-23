# Khisto

[![CI](https://github.com/khiops/khisto-python/actions/workflows/ci.yaml/badge.svg)](https://github.com/khiops/khisto-python/actions/workflows/ci.yaml)
[![Docs](https://github.com/khiops/khisto-python/actions/workflows/docs.yaml/badge.svg)](https://khiops.github.io/khisto-python/)
[![PyPI](https://img.shields.io/pypi/v/khisto)](https://pypi.org/project/khisto/)
[![Python](https://img.shields.io/pypi/pyversions/khisto)](https://pypi.org/project/khisto/)
[![License](https://img.shields.io/pypi/l/khisto)](LICENSE)

**Optimal Binning Histograms for Python**

Khisto is a Python library for creating histograms using the **Khiops optimal binning algorithm**. Unlike standard histograms that use fixed-width bins or simple heuristics, Khisto automatically determines the optimal number of bins and their variable widths to best represent the underlying data distribution.

Documentation is available at **[khiops.github.io/khisto-python](https://khiopsml.github.io/khisto-python/)**.

## Installation

```bash
pip install khisto
```

With matplotlib support:

```bash
pip install "khisto[matplotlib]"
```

## Quick Start

### NumPy-like API

```python
import numpy as np
from khisto import histogram

# Generate 10,000 samples from a standard Gaussian distribution.
data = np.random.normal(0, 1, 10000)

# Compute optimal histogram (drop-in replacement for np.histogram)
hist, bin_edges = histogram(data)
```

## Development

```bash
# Clone repository
git clone https://github.com/khiops/khisto-python.git
cd khisto-python

# Install with dev dependencies
uv sync --group dev --extra all

# Run tests
uv run pytest
```

## License

[BSD 3-Clause Clear License](LICENSE)

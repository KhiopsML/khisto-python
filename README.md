# Khisto

**Optimal Binning Histograms for Python**

Khisto is a Python library for creating histograms using the **Khiops optimal binning algorithm**. Unlike standard histograms that use fixed-width bins or simple heuristics, Khisto automatically determines the optimal number of bins and their variable widths to best represent the underlying data distribution.

## Features

- **Optimal Binning**: Uses the MODL (Minimum Description Length) principle to find the best discretization.
- **Variable-Width Bins**: Captures dense regions with fine bins and sparse regions with wider bins.
- **NumPy Compatible**: Drop-in replacement for `numpy.histogram`.
- **Matplotlib Integration**: `khisto.matplotlib.hist` works like `plt.hist`.
- **Minimal Dependencies**: Only requires NumPy (matplotlib optional for plotting).

| Standard Gaussian | Heavy-tailed Pareto |
| --- | --- |
| ![Adaptive Gaussian histogram](docs/images/gaussian-quick-start.png) | ![Adaptive Pareto histogram](docs/images/pareto-quick-start.png) |

## Reproducing The Example Distributions

The complete runnable script is available in `scripts/generate_distribution_examples.py`.

Run it from the repository root to regenerate both example distributions and the figure files used in this README:

```bash
python scripts/generate_distribution_examples.py
```

## Installation

```bash
pip install khisto
```

With matplotlib support:

```bash
pip install khisto[matplotlib]
```

## Quick Start

### NumPy-like API

```python
import numpy as np
from khisto import cumfreq, histogram

# Generate 10,000 samples from a standard Gaussian distribution.
data = np.random.normal(0, 1, 10000)

# Compute optimal histogram (drop-in replacement for np.histogram)
hist, bin_edges = histogram(data)

# With density normalization
density, bin_edges = histogram(data, density=True)

# Limit maximum number of bins
hist, bin_edges = histogram(data, max_bins=10)

# Specify range
hist, bin_edges = histogram(data, range=(-2, 2))

# Compute cumulative counts (SciPy-like cumfreq interface)
cumcount, bin_edges = cumfreq(data)

# Compute the cumulative distribution function
cdf, bin_edges = cumfreq(data, density=True)
```

Using 10,000 samples keeps the adaptive refinement visible while remaining fast to compute.

Heavy-tailed example:

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

# Generate 10,000 samples from a heavy-tailed Pareto distribution.
shape = 3
long_tail_data = np.random.pareto(shape, size=10000) + 1

# Plot an adaptive histogram on logarithmic axes.
n, bins, patches = hist(long_tail_data, density=True)
plt.xscale("log")
plt.yscale("log")
plt.show()
```

### Matplotlib Integration

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

# Generate 10,000 samples from a standard Gaussian distribution.
data = np.random.normal(0, 1, 10000)

# Density is usually the most interpretable view with variable-width bins.
n, bins, patches = hist(data, density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()

# Cumulative density follows matplotlib semantics.
n, bins, patches = hist(data, density=True, cumulative=True)
plt.ylabel('Cumulative probability')
plt.show()
```

## API Reference

### `khisto.histogram`

```python
def histogram(
    a: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
) -> tuple[ndarray, ndarray]
```

Compute an optimal histogram using the Khiops binning algorithm.

- `a`: Array-like input data. Nested sequences are flattened and concatenated into a single dataset.
- `range`: Optional lower and upper bounds. Values outside the interval are ignored.
- `max_bins`: Optional upper bound on the number of returned bins.
- `density`: Returns counts by default, or probability densities when set to `True`.

`khisto.histogram` is compatible in spirit with [`numpy.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html), but it does not support the `bins` or `weights` parameters.

### `khisto.cumfreq`

```python
def cumfreq(
    a: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
) -> tuple[ndarray, ndarray]
```

Compute a cumulative histogram using the Khiops binning algorithm.

- `a`: Array-like input data. Nested sequences are flattened and concatenated into a single dataset.
- `range`: Optional lower and upper bounds. Values outside the interval are ignored.
- `max_bins`: Optional upper bound on the number of returned bins.
- `density`: Returns cumulative counts by default, or cumulative probabilities when set to `True`.

`khisto.cumfreq` plays the same role as [`scipy.stats.cumfreq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cumfreq.html), but with adaptive variable-width bins and explicit `bin_edges` in the return value.

### `khisto.matplotlib.hist`

```python
def hist(
    x: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
    cumulative: bool | float = False,
    **kwargs,
) -> tuple[ndarray, ndarray, Any]
```

Plot an optimal histogram using matplotlib.

- `x`: Array-like input data, or a sequence of array-like objects. Sequences are concatenated and histogrammed as a single dataset.
- `max_bins`: Optional upper bound on the number of bins computed by Khisto.
- `density`: Returns counts by default. For variable-width bins, `density=True` is often easier to interpret visually.
- `cumulative`: Matches `matplotlib.pyplot.hist`. With `density=True`, the returned values are cumulative probabilities and the last bin equals 1.
- `**kwargs`: Other plotting options such as `histtype`, `orientation`, `log`, `color`, `label`, or `ax`.

Like [`matplotlib.pyplot.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html), this function returns `(n, bins, patches)`. The `bins`, `weights`, and `stacked` arguments are not supported.

## How It Works

Khisto uses the Khiops optimal binning algorithm based on the MODL (Minimum Optimal Description Length) principle. Instead of using fixed-width bins like traditional histograms, it:

1. Analyzes the data distribution
2. Finds bin boundaries that minimize information loss
3. Creates variable-width bins that adapt to data density

This results in histograms that better represent the underlying distribution, with finer bins in dense regions and wider bins in sparse regions.

The method implemented in Khiops is comprehensively detailed in [2] and further extended in [1].

- [1] M. Boullé. Floating-point histograms for exploratory analysis of large scale real-world data sets. Intelligent Data Analysis, 28(5):1347-1394, 2024
- [2] V. Zelaya Mendizábal, M. Boullé, F. Rossi. Fast and fully-automated histograms for large-scale data sets. Computational Statistics & Data Analysis, 180:0-0, 2023

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

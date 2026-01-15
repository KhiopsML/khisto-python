# Khisto

**Optimal Binning Histograms for Python**

Khisto is a Python library for creating histograms using the **Khiops optimal binning algorithm**. Unlike standard histograms that use fixed-width bins or simple heuristics, Khisto automatically determines the optimal number of bins and their variable widths to best represent the underlying data distribution.

## Features

- **Optimal Binning**: Uses the MODL (Minimum Description Length) principle to find the best discretization.
- **Variable-Width Bins**: Captures dense regions with fine bins and sparse regions with wider bins.
- **NumPy Compatible**: Drop-in replacement for `numpy.histogram`.
- **Matplotlib Integration**: `khisto.matplotlib.hist` works like `plt.hist`.
- **Minimal Dependencies**: Only requires NumPy (matplotlib optional for plotting).

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

# Generate data
data = np.random.normal(0, 1, 1000)

# Compute optimal histogram (drop-in replacement for np.histogram)
hist, bin_edges = histogram(data)

# With density normalization
density, bin_edges = histogram(data, density=True)

# Limit maximum number of bins
hist, bin_edges = histogram(data, max_bins=10)

# Specify range
hist, bin_edges = histogram(data, range=(-2, 2))
```

### Matplotlib Integration

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

data = np.random.normal(0, 1, 1000)

# Create optimal histogram plot
n, bins, patches = hist(data)
plt.show()

# With density
n, bins, patches = hist(data, density=True)
plt.xlabel('Value')
plt.ylabel('Density')
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

**Parameters:**
- `a`: Input data. The histogram is computed over the flattened array.
- `range`: The lower and upper range of the bins. Values outside are ignored.
- `max_bins`: Maximum number of bins. If None, the algorithm selects optimal.
- `density`: If True, return probability density. If False, return counts.

**Returns:**
- `hist`: The values of the histogram (counts or density).
- `bin_edges`: The bin edges (length = len(hist) + 1).

### `khisto.matplotlib.hist`

```python
def hist(
    x: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
    histtype: str = "bar",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    log: bool = False,
    color: Optional[str] = None,
    label: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> tuple[ndarray, ndarray, Any]
```

Plot an optimal histogram using matplotlib.

**Parameters:**
- `x`: Input data.
- `range`: The lower and upper range of the bins.
- `max_bins`: Maximum number of bins.
- `density`: If True, plot probability density.
- `histtype`: Type of histogram (`"bar"`, `"step"`, `"stepfilled"`).
- `orientation`: `"vertical"` or `"horizontal"`.
- `log`: If True, set log scale on the value axis.
- `ax`: Matplotlib axes to plot on.

**Returns:**
- `n`: The histogram values.
- `bins`: The bin edges.
- `patches`: The matplotlib patches.

## How It Works

Khisto uses the Khiops optimal binning algorithm based on the MODL (Minimum Optimal Description Length) principle. Instead of using fixed-width bins like traditional histograms, it:

1. Analyzes the data distribution
2. Finds bin boundaries that minimize information loss
3. Creates variable-width bins that adapt to data density

This results in histograms that better represent the underlying distribution, with finer bins in dense regions and wider bins in sparse regions.

## Development

```bash
# Clone repository
git clone https://github.com/khiops/khisto-python.git
cd khisto-python

# Install with dev dependencies
pip install -e ".[matplotlib]"

# Run tests
pytest
```

## License

[BSD 3-Clause Clear License](LICENSE)

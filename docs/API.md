# Khisto API Reference

Complete API reference for the Khisto library.

## Table of Contents

- [Array API](#array-api)
  - [histogram](#histogram)
  - [cumfreq](#cumfreq)
- [Core API](#core-api)
  - [compute_histogram](#compute_histogram)
  - [HistogramResult](#histogramresult)
- [Matplotlib API](#matplotlib-api)
  - [hist](#hist)
- [How It Works](#how-it-works)

---

## Array API

### `histogram`

```python
khisto.histogram(
    a: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]
```

Compute an optimal histogram using the Khiops binning algorithm.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a` | `ArrayLike` | required | Input data. Nested sequences are flattened and concatenated into a single dataset. |
| `range` | `tuple[float, float]` | `None` | Lower and upper range of the bins. Values outside are ignored. |
| `max_bins` | `int` | `None` | Maximum number of bins. If not provided, the optimal number is determined automatically. |
| `density` | `bool` | `False` | If `False`, return counts; if `True`, return probability density values. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `hist` | `NDArray[np.floating]` | The values of the histogram. |
| `bin_edges` | `NDArray[np.floating]` | Array of length `len(hist) + 1` containing the bin edges. |

#### See Also

- [`numpy.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) — NumPy's histogram function (`bins` and `weights` are not supported in Khisto).

#### Examples

Basic usage:

```python
import numpy as np
from khisto import histogram

data = np.random.normal(0, 1, 1000)

# Compute histogram
hist, bin_edges = histogram(data)
print(f"Number of bins: {len(hist)}")
print(f"Bin edges: {bin_edges}")
```

With density normalization:

```python
density, bin_edges = histogram(data, density=True)
# Verify normalization: integral should be ~1
widths = np.diff(bin_edges)
print(f"Integral: {np.sum(density * widths)}")  # ~1.0
```

Limiting maximum bins:

```python
hist, bin_edges = histogram(data, max_bins=5)
print(f"Number of bins: {len(hist)}")  # <= 5
```

Concatenating nested inputs into a single dataset:

```python
data = [np.array([0.0, 1.0]), np.array([2.0, 3.0, 4.0])]
hist, bin_edges = histogram(data)
print(hist.sum())  # 5
```

---

## Core API

The core API provides direct access to the Khiops histogram computation with detailed output.

### `compute_histogram`

```python
khisto.core.compute_histogram(
    x: ArrayLike,
) -> list[HistogramResult]
```

Compute optimal histograms at all granularity levels using the Khiops binning algorithm.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ArrayLike` | required | Input data array. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `results` | `list[HistogramResult]` | List of `HistogramResult` objects for all granularity levels, from coarsest to finest. |

#### See Also

- [`khisto.histogram`](#histogram) — Simplified interface returning `(hist, bin_edges)`.

#### Examples

Basic usage:

```python
import numpy as np
from khisto.core import compute_histogram

data = np.random.normal(0, 1, 1000)
results = compute_histogram(data)

# Find the optimal histogram
for r in results:
    if r.is_best:
        print(f"Optimal: {len(r.frequency)} bins")
        print(f"Bin edges: {r.bin_edges}")
```

---

### `HistogramResult`

```python
@dataclass
class HistogramResult:
    lower_bound: NDArray[np.floating]
    upper_bound: NDArray[np.floating]
    frequency: NDArray[np.int64]
    probability: NDArray[np.floating]
    density: NDArray[np.floating]
    is_best: bool
    granularity: int
```

A structured result containing all histogram information.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `lower_bound` | `NDArray[np.floating]` | Lower bounds of each bin. |
| `upper_bound` | `NDArray[np.floating]` | Upper bounds of each bin. |
| `frequency` | `NDArray[np.int64]` | Count of samples in each bin. |
| `probability` | `NDArray[np.floating]` | Probability mass in each bin (frequency / total). |
| `density` | `NDArray[np.floating]` | Probability density (probability / bin_width). |
| `is_best` | `bool` | Whether this is the optimal histogram. |
| `granularity` | `int` | Granularity level (number of bins at this level). |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `bin_edges` | `NDArray[np.floating]` | Array of bin edges (length = n_bins + 1). |
| `bin_widths` | `NDArray[np.floating]` | Width of each bin. |
| `bin_centers` | `NDArray[np.floating]` | Center of each bin. |

#### Examples

```python
import numpy as np
from khisto.core import compute_histogram

data = np.random.normal(0, 1, 1000)
results = compute_histogram(data)
result = next(r for r in results if r.is_best)

# Access bin information
print(f"Bin edges: {result.bin_edges}")
print(f"Bin widths: {result.bin_widths}")
print(f"Bin centers: {result.bin_centers}")

# Access histogram values
print(f"Frequencies: {result.frequency}")
print(f"Probabilities: {result.probability}")
print(f"Densities: {result.density}")

# Check optimality
print(f"Is best: {result.is_best}")
print(f"Granularity: {result.granularity}")
```

---

## Matplotlib API

### `hist`

```python
khisto.matplotlib.hist(
    x: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
    cumulative: bool | float = False,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating], Any]
```

Compute and plot an optimal histogram.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ArrayLike` | required | Input data, or a sequence of array-like objects. Nested inputs are concatenated and histogrammed as one dataset. |
| `max_bins` | `int` | `None` | Maximum number of bins. If `None`, uses optimal binning. |
| `density` | `bool` | `False` | If `True`, return and plot probability densities. If `False`, return counts. |
| `cumulative` | `bool or float` | `False` | Cumulative mode, following `matplotlib.pyplot.hist`. Negative values accumulate in reverse order. |

Other parameters are passed to matplotlib for styling. `ax` can be provided to draw on a specific axes. The `bins`, `weights`, and `stacked` arguments are not supported.

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `n` | `NDArray[np.floating]` | The values of the histogram bins (probability density by default). |
| `bins` | `NDArray[np.floating]` | The bin edges. |
| `patches` | `Any` | Container of individual artists (bars or StepPatch). |

#### See Also

- [`matplotlib.pyplot.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) — Matplotlib's histogram function.
- [`khisto.cumfreq`](#cumfreq) — Array-level cumulative histogram API.
- [`khisto.histogram`](#histogram) — Underlying non-cumulative histogram computation.

#### Examples

Basic plot:

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

data = np.random.normal(0, 1, 10000)

# Density is usually the clearest view with variable-width bins.
n, bins, patches = hist(data, density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Optimal Histogram')
plt.show()
```

Cumulative density:

```python
n, bins, patches = hist(data, density=True, cumulative=True)
plt.ylabel('Cumulative probability')
plt.show()
```

Heavy-tailed Pareto example:

```python
shape = 3
long_tail_data = np.random.pareto(shape, size=10000) + 1

n, bins, patches = hist(long_tail_data, density=True)
plt.xscale('log')
plt.yscale('log')
plt.show()
```

---

## How It Works

Khisto uses the Khiops optimal binning algorithm based on the MODL (Minimum Optimal Description Length) principle. Instead of using fixed-width bins like traditional histograms, it:

1. Analyzes the data distribution
2. Finds bin boundaries that minimize information loss
3. Creates variable-width bins that adapt to data density

This results in histograms that better represent the underlying distribution, with finer bins in dense regions and wider bins in sparse regions.

The method implemented in Khiops is comprehensively detailed in [2] and further extended in [1].

- [1] M. Boullé. Floating-point histograms for exploratory analysis of large scale real-world data sets. Intelligent Data Analysis, 28(5):1347-1394, 2024
- [2] V. Zelaya Mendizábal, M. Boullé, F. Rossi. Fast and fully-automated histograms for large-scale data sets. Computational Statistics & Data Analysis, 180:0-0, 2023

---

## Type Aliases

```python
ArrayLike = Union[list, np.ndarray, ...]
```

Any array-like object that can be converted to a NumPy array.

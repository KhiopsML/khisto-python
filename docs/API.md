# Khisto API Reference

Complete API reference for the Khisto library.

## Table of Contents

- [Array API](#array-api)
  - [histogram](#histogram)
- [Core API](#core-api)
  - [compute_histogram](#compute_histogram)
  - [HistogramResult](#histogramresult)
- [Matplotlib API](#matplotlib-api)
  - [hist](#hist)

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

Drop-in replacement for [`numpy.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html), using optimal binning instead of fixed-width bins.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a` | `ArrayLike` | required | Input data. The histogram is computed over the flattened array. |
| `range` | `tuple[float, float]` | `None` | Lower and upper range of the bins. Values outside are ignored. |
| `max_bins` | `int` | `None` | Maximum number of bins. If not provided, the optimal number is determined automatically. |
| `density` | `bool` | `False` | If `True`, return probability density values; otherwise return counts. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `hist` | `NDArray[np.floating]` | The values of the histogram. |
| `bin_edges` | `NDArray[np.floating]` | Array of length `len(hist) + 1` containing the bin edges. |

#### See Also

- [`numpy.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) — NumPy's standard histogram function.

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
result = compute_histogram(data)

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
    ax: Optional[Axes] = None,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating], Any]
```

Compute and plot an optimal histogram.

Drop-in replacement for [`matplotlib.pyplot.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) using Khisto's optimal binning algorithm.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ArrayLike` | required | Input data. |
| `range` | `tuple[float, float]` | `None` | Lower and upper range of the bins. Values outside are ignored. |
| `max_bins` | `int` | `None` | Maximum number of bins. If `None`, uses optimal binning. |
| `ax` | `Axes` | `None` | Axes to plot on. If `None`, uses current axes. |
| `**kwargs` | | | Other parameters passed to matplotlib. See [`matplotlib.pyplot.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html). |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `n` | `NDArray[np.floating]` | The values of the histogram bins. |
| `bins` | `NDArray[np.floating]` | The bin edges. |
| `patches` | `Any` | Container of individual artists (bars or StepPatch). |

#### See Also

- [`matplotlib.pyplot.hist`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html) — Full documentation of supported parameters.
- [`khisto.histogram`](#histogram) — Underlying histogram computation.

#### Examples

Basic plot:

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

data = np.random.normal(0, 1, 1000)

n, bins, patches = hist(data)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title('Optimal Histogram')
plt.show()
```

Density plot:

```python
n, bins, patches = hist(data, density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
```

Step histogram:

```python
n, bins, patches = hist(data, histtype='step', color='blue', label='Data')
plt.legend()
plt.show()
```

Using specific axes:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

hist(data, ax=ax1)
ax1.set_title('Counts')

hist(data, density=True, ax=ax2)
ax2.set_title('Density')

plt.tight_layout()
plt.show()
```

---

## Type Aliases

```python
ArrayLike = Union[list, np.ndarray, ...]
```

Any array-like object that can be converted to a NumPy array.

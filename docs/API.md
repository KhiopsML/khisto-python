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

This function is designed as a drop-in replacement for `numpy.histogram`, providing automatic optimal binning instead of fixed-width bins.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a` | `ArrayLike` | required | Input data. The histogram is computed over the flattened array. |
| `range` | `tuple[float, float]` | `None` | The lower and upper range of the bins. If not provided, range is `(a.min(), a.max())`. Values outside the range are ignored. |
| `max_bins` | `int` | `None` | Maximum number of bins. If not provided, the algorithm determines the optimal number. |
| `density` | `bool` | `False` | If `False`, the result contains the number of samples in each bin. If `True`, the result is the value of the probability density function at the bin, normalized such that the integral over the range is 1. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `hist` | `NDArray[np.floating]` | The values of the histogram. |
| `bin_edges` | `NDArray[np.floating]` | Array of length `len(hist) + 1` containing the bin edges. |

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
    max_bins: Optional[int] = None,
    range: Optional[tuple[float, float]] = None,
    return_all: bool = False,
) -> Union[HistogramResult, list[HistogramResult]]
```

Compute an optimal histogram using the Khiops binning algorithm.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ArrayLike` | required | Input data array. |
| `max_bins` | `int` | `None` | Maximum number of bins for the histogram. |
| `range` | `tuple[float, float]` | `None` | The range `(min, max)` over which to compute the histogram. Values outside this range are ignored. |
| `return_all` | `bool` | `False` | If `True`, return all granularity levels computed by the algorithm. If `False`, return only the optimal histogram. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `result` | `HistogramResult` or `list[HistogramResult]` | If `return_all=False`, returns a single `HistogramResult` for the optimal histogram. If `return_all=True`, returns a list of `HistogramResult` objects for all granularity levels. |

#### Examples

Basic usage:

```python
import numpy as np
from khisto.core import compute_histogram

data = np.random.normal(0, 1, 1000)
result = compute_histogram(data)

print(f"Number of bins: {len(result.frequency)}")
print(f"Bin edges: {result.bin_edges}")
print(f"Is optimal: {result.is_best}")
```

Get all granularity levels:

```python
results = compute_histogram(data, return_all=True)
for r in results:
    marker = " (best)" if r.is_best else ""
    print(f"Granularity {r.granularity}: {len(r.frequency)} bins{marker}")
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
    histtype: str = "bar",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    log: bool = False,
    color: Optional[str] = None,
    label: Optional[str] = None,
    ax: Optional[Axes] = None,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating], Any]
```

Plot an optimal histogram using matplotlib.

This function is designed to work similarly to `matplotlib.pyplot.hist`, but uses Khiops optimal binning.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `ArrayLike` | required | Input data. |
| `range` | `tuple[float, float]` | `None` | The lower and upper range of the bins. |
| `max_bins` | `int` | `None` | Maximum number of bins. |
| `density` | `bool` | `False` | If `True`, plot probability density instead of counts. |
| `histtype` | `str` | `"bar"` | Type of histogram: `"bar"`, `"step"`, or `"stepfilled"`. |
| `orientation` | `str` | `"vertical"` | `"vertical"` or `"horizontal"`. |
| `log` | `bool` | `False` | If `True`, set log scale on the value axis. |
| `color` | `str` | `None` | Color of the histogram. |
| `label` | `str` | `None` | Label for legend. |
| `ax` | `Axes` | `None` | Matplotlib axes to plot on. If `None`, uses current axes. |
| `**kwargs` | | | Additional arguments passed to matplotlib's `bar` or `stairs`. |

#### Returns

| Return | Type | Description |
|--------|------|-------------|
| `n` | `NDArray[np.floating]` | The values of the histogram bins. |
| `bins` | `NDArray[np.floating]` | The bin edges. |
| `patches` | `Any` | Container of individual artists (bars or StepPatch). |

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

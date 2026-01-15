# API Comparison

This document compares the Khisto API with NumPy and Matplotlib, highlighting similarities and differences.

## NumPy Comparison

### `numpy.histogram` vs `khisto.histogram`

Khisto's `histogram` function is designed as a drop-in replacement for `numpy.histogram`.

#### Signature Comparison

```python
# NumPy
numpy.histogram(
    a,
    bins=10,
    range=None,
    density=None,
    weights=None,
)

# Khisto
khisto.histogram(
    a,
    range=None,
    max_bins=None,
    density=False,
)
```

#### Key Differences

| Feature | NumPy | Khisto |
|---------|-------|--------|
| **Binning method** | Fixed-width bins | Optimal variable-width bins |
| **Bins parameter** | `bins` (int or edges) | `max_bins` (optional limit) |
| **Default bins** | 10 fixed bins | Auto-determined optimal |
| **Weights support** | Yes | No |
| **Returns** | `(hist, bin_edges)` | `(hist, bin_edges)` |

#### Usage Comparison

```python
import numpy as np
from khisto import histogram

data = np.random.normal(0, 1, 1000)

# NumPy - fixed 10 bins
np_hist, np_edges = np.histogram(data)

# Khisto - optimal bins (automatic)
khisto_hist, khisto_edges = histogram(data)

# NumPy - specified bin count
np_hist, np_edges = np.histogram(data, bins=20)

# Khisto - maximum bin count
khisto_hist, khisto_edges = histogram(data, max_bins=20)

# Both support density normalization
np_density, _ = np.histogram(data, density=True)
khisto_density, _ = histogram(data, density=True)

# Both support range specification
np_hist, _ = np.histogram(data, range=(-2, 2))
khisto_hist, _ = histogram(data, range=(-2, 2))
```

#### When to Use Each

| Use NumPy | Use Khisto |
|-----------|------------|
| Need fixed-width bins | Want optimal data representation |
| Need weighted histograms | Want automatic bin selection |
| Need specific bin edges | Want adaptive bin widths |
| Performance-critical loops | Data visualization |

---

## Matplotlib Comparison

### `matplotlib.pyplot.hist` vs `khisto.matplotlib.hist`

Khisto's `hist` function works similarly to matplotlib's `hist`, but with optimal binning.

#### Signature Comparison

```python
# Matplotlib
matplotlib.pyplot.hist(
    x,
    bins=10,
    range=None,
    density=False,
    weights=None,
    cumulative=False,
    bottom=None,
    histtype='bar',
    align='mid',
    orientation='vertical',
    rwidth=None,
    log=False,
    color=None,
    label=None,
    stacked=False,
    **kwargs,
)

# Khisto
khisto.matplotlib.hist(
    x,
    range=None,
    max_bins=None,
    density=False,
    histtype='bar',
    orientation='vertical',
    log=False,
    color=None,
    label=None,
    ax=None,
    **kwargs,
)
```

#### Key Differences

| Feature | Matplotlib | Khisto |
|---------|------------|--------|
| **Binning** | Fixed-width | Optimal variable-width |
| **Bins param** | `bins` | `max_bins` |
| **Axes param** | Implicit (current) | Explicit `ax` parameter |
| **Cumulative** | Supported | Not supported |
| **Stacked** | Supported | Not supported |
| **Weights** | Supported | Not supported |
| **Multiple datasets** | Supported | Single dataset only |

#### Usage Comparison

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

data = np.random.normal(0, 1, 1000)

# Matplotlib - fixed bins
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.hist(data, bins=30)
ax1.set_title('Matplotlib (30 bins)')

hist(data, ax=ax2)
ax2.set_title('Khisto (optimal bins)')

plt.tight_layout()
plt.show()
```

#### Common Parameters (Same Behavior)

```python
# Both support these parameters identically:

# density normalization
plt.hist(data, density=True)
hist(data, density=True)

# histogram type
plt.hist(data, histtype='step')
hist(data, histtype='step')

# orientation
plt.hist(data, orientation='horizontal')
hist(data, orientation='horizontal')

# log scale
plt.hist(data, log=True)
hist(data, log=True)

# color and label
plt.hist(data, color='blue', label='Data')
hist(data, color='blue', label='Data')
```

---

## Migration Guide

### From NumPy

```python
# Before (NumPy)
import numpy as np
hist, edges = np.histogram(data, bins=30)

# After (Khisto)
from khisto import histogram
hist, edges = histogram(data, max_bins=30)  # max_bins is optional
```

### From Matplotlib

```python
# Before (Matplotlib)
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(data, bins=30)

# After (Khisto)
from khisto.matplotlib import hist
n, bins, patches = hist(data, max_bins=30)  # max_bins is optional
```

---

## Feature Matrix

| Feature | NumPy | Matplotlib | Khisto |
|---------|-------|------------|--------|
| Fixed-width bins | ✓ | ✓ | ✗ |
| Optimal bins | ✗ | ✗ | ✓ |
| Variable-width bins | Manual | Manual | Auto |
| Density | ✓ | ✓ | ✓ |
| Range | ✓ | ✓ | ✓ |
| Weights | ✓ | ✓ | ✗ |
| Cumulative | ✗ | ✓ | ✗ |
| Plotting | ✗ | ✓ | ✓ |
| Step histogram | ✗ | ✓ | ✓ |
| Horizontal | ✗ | ✓ | ✓ |
| Log scale | ✗ | ✓ | ✓ |

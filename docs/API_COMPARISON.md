# API Comparison

This document compares the Khisto API with standard Python data visualization and array libraries. Khisto is designed to be a drop-in replacement or a close alternative to these libraries, with the key difference being its **optimal, variable-width binning algorithm**.

## Numpy vs Khisto Array API

The `khisto.array` module is designed to be familiar to users of `numpy.histogram`.

| Feature | `numpy.histogram` | `khisto.array.histogram` |
| :--- | :--- | :--- |
| **Binning Strategy** | Fixed width (default 10) or simple heuristics (e.g., 'auto', 'fd') | **Optimal variable-width** (MODL algorithm) |
| **Bin Parameter** | `bins` (int, sequence, or string) | `granularity` (int, "best", or None) |
| **Output** | `(hist, bin_edges)` | `(values, bin_edges)` |
| **Density** | `density=True/False` | `density=True/False` |

### Key Differences
1.  **Variable Widths**: Numpy's `bins='auto'` still produces equal-width bins. Khisto produces bins of varying widths to better capture data density.
2.  **Granularity**: Instead of specifying a number of bins, you specify a `granularity` level. `granularity="best"` automatically selects the most informative level.

### Example

```python
import numpy as np
import khisto.array as kha

data = np.random.normal(0, 1, 1000)

# Numpy
hist, edges = np.histogram(data, bins='auto', density=True)

# Khisto
values, edges = kha.histogram(data, granularity='best', density=True)
```

---

## Matplotlib vs Khisto Matplotlib API

The `khisto.matplotlib` module provides functions that integrate directly with `matplotlib.pyplot`.

| Feature | `plt.hist` | `khisto.matplotlib.hist` |
| :--- | :--- | :--- |
| **Binning** | `bins` argument (default 10) | `granularity` argument (default "best") |
| **Bar Widths** | Usually equal | **Variable** (data-dependent) |
| **Return Value** | `(n, bins, patches)` | `(n, bins, patches)` (compatible) |
| **Visuals** | Standard bars | Bars with varying widths |

### Key Differences
1.  **Drop-in Replacement**: `khisto.matplotlib.hist` mimics the signature and return values of `plt.hist`, making it easy to swap.
2.  **Dedicated Function**: Khisto also offers `khisto.matplotlib.histogram`, which returns just the `BarContainer` (or list of containers), which is often cleaner for object-oriented matplotlib usage.

### Example

```python
import matplotlib.pyplot as plt
import khisto.matplotlib as khm

# Standard Matplotlib
plt.hist(data, bins=30)

# Khisto (Drop-in replacement)
khm.hist(data)

# Khisto (Object-Oriented Style)
fig, ax = plt.subplots()
khm.histogram(x=data, ax=ax)
```

---

## Plotly vs Khisto Plotly API

The `khisto.plotly` module mimics the high-level interface of Plotly Express (`px`).

| Feature | `px.histogram` | `khisto.plotly.histogram` |
| :--- | :--- | :--- |
| **Binning** | `nbins` argument | `granularity` argument |
| **Engine** | Plotly Express | Plotly Graph Objects (returns `go.Figure`) |
| **Interface** | DataFrame + column names | DataFrame + column names (compatible) |
| **Visuals** | Equal-width bars | Variable-width bars |

### Key Differences
1.  **API Mimicry**: Khisto copies the argument names (`data_frame`, `x`, `color`, `facet_col`, etc.) of Plotly Express, so you can often just change the import.
2.  **Implementation**: While `px.histogram` relies on the plotting engine to do the binning, `khisto.plotly.histogram` pre-calculates the optimal bins and draws them explicitly.

### Example

```python
import plotly.express as px
import khisto.plotly as khp

df = px.data.tips()

# Plotly Express
fig = px.histogram(df, x="total_bill", color="sex", nbins=20)
fig.show()

# Khisto
fig = khp.histogram(df, x="total_bill", color="sex")
fig.show()
```

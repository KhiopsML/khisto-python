# Khisto API Reference

This document provides a reference for the Khisto library API, organized by module.

## Array API (`khisto.array`)

The `khisto.array` module provides core functions for computing optimal histograms and cumulative distributions using the Khiops algorithm. These functions return raw data (arrays or DataFrames) rather than plots.

### Histogram Functions

#### `histogram`

```python
def histogram(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
    density: bool = True,
) -> Union[tuple[ArrayT, ArrayT], list[tuple[ArrayT, ArrayT]]]
```

Compute histogram using optimal binning.

**Parameters:**
- `x`: Input data (Array API compliant array, list/tuple, or Narwhals Series).
- `granularity`: Granularity level to use (`"best"`, `int`, or `None`).
- `density`: If `True`, return probability density values. If `False`, return frequency counts.

**Returns:**
- If `granularity` is `"best"` or `int`: A tuple `(values, bin_edges)`.
- If `granularity` is `None`: A list of `(values, bin_edges)` tuples for all granularities.

#### `histogram_bin_edges`

```python
def histogram_bin_edges(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
) -> Union[ArrayT, list[ArrayT]]
```

Compute histogram bin edges using optimal binning.

**Parameters:**
- `x`: Input data.
- `granularity`: Granularity level (`"best"`, `int`, or `None`).

**Returns:**
- Array of bin edges (or list of arrays if `granularity` is `None`).

#### `histogram_table`

```python
def histogram_table(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = None,
) -> nw.DataFrame
```

Return detailed histogram information as a DataFrame.

**Parameters:**
- `x`: Input data.
- `granularity`: Granularity level (`"best"`, `int`, or `None`).

**Returns:**
- A DataFrame with columns: `lower_bound`, `upper_bound`, `length`, `frequency`, `probability`, `density`, `center`, `granularity`, `is_best`.

### Cumulative Distribution Functions

#### `ecdf`

```python
def ecdf(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
) -> Union[ECDFResult, ECDFResultCollection]
```

Compute an empirical CDF that can be evaluated at any point.

**Parameters:**
- `x`: Input data.
- `granularity`: Granularity level (`"best"`, `int`, or `None`).

**Returns:**
- An `ECDFResult` object (callable) or `ECDFResultCollection`.

#### `ecdf_values`

```python
def ecdf_values(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
    density: bool = True,
) -> Union[tuple[ArrayT, ArrayT], list[tuple[ArrayT, ArrayT]]]
```

Compute discrete ECDF values at bin edges.

**Parameters:**
- `x`: Input data.
- `granularity`: Granularity level (`"best"`, `int`, or `None`).
- `density`: If `True`, return cumulative probability (0.0-1.0). If `False`, return cumulative counts.

**Returns:**
- Tuple of `(cdf_values, positions)` (or list of tuples).

#### `ecdf_values_table`

```python
def ecdf_values_table(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = None,
) -> nw.DataFrame
```

Return ECDF values as a DataFrame.

**Parameters:**
- `x`: Input data.
- `granularity`: Granularity level (`"best"`, `int`, or `None`).

**Returns:**
- A DataFrame with columns: `position`, `cumulative_probability`, `cumulative_frequency`, `granularity`, `is_best`.

---

## Matplotlib API (`khisto.matplotlib`)

The `khisto.matplotlib` module provides matplotlib-compatible plotting functions that use Khisto's optimal binning.

#### `histogram`

```python
def histogram(
    data: Optional[IntoDataFrame] = None,
    *,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    hue: Optional[str] = None,
    ax: Optional[Axes] = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    granularity: Optional[GranularityT] = "best",
    density: bool = True,
    # ... standard matplotlib args (alpha, color, etc.)
) -> BarContainer | list[BarContainer]
```

Create a histogram using Khisto's optimal binning algorithm.

**Parameters:**
- `data`: DataFrame-like object.
- `x`: Column name or array-like data.
- `hue`: Column name for grouping.
- `ax`: Matplotlib axes.
- `orientation`: 'vertical' or 'horizontal'.
- `granularity`: Granularity level.
- `density`: Plot density (default True) or counts.

**Returns:**
- `BarContainer` or list of `BarContainer` objects.

#### `ecdf`

```python
def ecdf(
    data: Optional[IntoDataFrame] = None,
    *,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    hue: Optional[str] = None,
    ax: Optional[Axes] = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    granularity: Optional[GranularityT] = "best",
    density: bool = True,
    # ... standard matplotlib args
) -> Line2D | list[Line2D] | None
```

Create a cumulative distribution plot using Khisto's optimal binning algorithm.

**Parameters:**
- `data`: DataFrame-like object.
- `x`: Column name or array-like data.
- `hue`: Column name for grouping.
- `ax`: Matplotlib axes.
- `orientation`: 'vertical' or 'horizontal'.
- `granularity`: Granularity level.
- `density`: Plot probability (0-1) or frequency counts.

**Returns:**
- `Line2D` or list of `Line2D` objects.

#### `hist`

```python
def hist(
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    bins: Optional[GranularityT] = None,
    # ... standard matplotlib.pyplot.hist args
    *,
    data: Optional[IntoDataFrame] = None,
    hue: Optional[str] = None,
    granularity: Optional[GranularityT] = "best",
    # ...
) -> Union[tuple[Any, Any, Any], Any]
```

Compute and plot a histogram or cumulative distribution. Combines functionality of `histogram` and `ecdf` with an interface similar to `matplotlib.pyplot.hist`.

---

## Plotly API (`khisto.plotly`)

The `khisto.plotly` module provides Plotly Express-compatible plotting functions.

#### `histogram`

```python
def histogram(
    data_frame: Optional[IntoDataFrame] = None,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    color: Optional[str] = None,
    # ... standard plotly express args
    granularity: Optional[GranularityT] = "best",
    # ...
) -> go.Figure
```

Create a histogram using Khisto's optimal binning algorithm.

**Parameters:**
- `data_frame`: DataFrame-like object.
- `x`: Column name or array-like data.
- `color`: Column name for grouping.
- `granularity`: Granularity level.

**Returns:**
- `plotly.graph_objects.Figure`

#### `ecdf`

```python
def ecdf(
    data_frame: Optional[IntoDataFrame] = None,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    # ... standard plotly express args
    granularity: Optional[GranularityT] = "best",
    # ...
) -> go.Figure
```

Create a cumulative distribution plot using Khisto's optimal binning algorithm.

**Parameters:**
- `data_frame`: DataFrame-like object.
- `x`: Column name or array-like data.
- `granularity`: Granularity level.

**Returns:**
- `plotly.graph_objects.Figure`

#### `ridgeplot`

```python
def ridgeplot(
    data_frame: Any,
    x: str,
    y: str,
    granularity: Optional[GranularityT] = "best",
    # ... styling args (opacity, overlap, etc.)
) -> go.Figure
```

Create a ridge plot (joy plot) using Khisto's optimal binning algorithm.

**Parameters:**
- `data_frame`: DataFrame-like object.
- `x`: Column name for the value axis (horizontal).
- `y`: Column name for the category axis (vertical).
- `granularity`: Granularity level.
- `overlap`: Amount of overlap between ridges (0.0 to 1.0).

**Returns:**
- `plotly.graph_objects.Figure`

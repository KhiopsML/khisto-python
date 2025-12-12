"""Matplotlib cumulative distribution visualization with optimal binning.

This module provides a matplotlib-compatible cumulative distribution function
that uses Khisto's optimal binning algorithm for automatic bin selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union, Any

import pyarrow as pa
import narwhals as nw

from khisto.array import ecdf_values_table
from khisto.utils._compat._optional import import_optional_dependency, Extras

import_optional_dependency("matplotlib", extra=Extras.MATPLOTLIB, errors="raise")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoDataFrame, IntoSeries


# ============================================================================
# Helper Functions
# ============================================================================


def _process_input_data(
    data: Optional[IntoDataFrame],
    x: Optional[Union[str, ArrayT, IntoSeries]],
) -> tuple[nw.DataFrame[Any] | nw.LazyFrame[Any], str]:
    """Process input data into a Narwhals DataFrame.

    Parameters
    ----------
    data : IntoDataFrame or None
        DataFrame-like object or None
    x : str, ArrayT, IntoSeries, or None
        Column name or array-like data

    Returns
    -------
    tuple[nw.DataFrame, str]
        Processed DataFrame and column name for x values
    """
    if data is None and x is None:
        raise ValueError("Either 'data' or 'x' must be provided")

    if data is None:
        # Direct array input
        if isinstance(x, str):
            raise ValueError("When data is None, x must be an array, not a column name")
        # Wrap array in a DataFrame using from_dict with pyarrow backend
        df = nw.from_dict({"_value": x}, backend=pa)
        x_column = "_value"
    else:
        # DataFrame input - let narwhals handle it
        df = nw.from_native(data)
        if x is None:
            raise ValueError("When data is provided, x column name must be specified")
        if not isinstance(x, str):
            raise ValueError(
                "When 'data' is provided, 'x' must be a column name (string), not an array. "
                "Either pass x as a column name, or use cumulative(x=array) without the data parameter."
            )
        x_column = x

    return df, x_column


def _compute_cumulative_data(
    df: nw.DataFrame[Any] | nw.LazyFrame[Any],
    x_column: str,
    hue: Optional[str],
    granularity: Optional[GranularityT],
) -> dict[Any, nw.DataFrame[Any]]:
    """Compute cumulative distribution data for each group.

    Parameters
    ----------
    df : nw.DataFrame
        Input DataFrame
    x_column : str
        Name of column containing values
    hue : str or None
        Column name for grouping/coloring
    granularity : int, 'best', or None
        Granularity level for binning

    Returns
    -------
    dict[Any, nw.DataFrame]
        Dictionary mapping group keys to their cumulative line DataFrames
    """
    # If no grouping, treat as single group
    if hue is None:
        groups = {None: df}
    else:
        # Group by hue column
        unique_values = df[hue].unique(maintain_order=True).to_list()
        groups = {val: df.filter(nw.col(hue) == val) for val in unique_values}

    # Compute cumulative data for each group
    cumulative_data = {}
    for group_key, group_df in groups.items():
        # Use ecdf_values_table to get cumulative distribution
        cumulative_df = ecdf_values_table(
            group_df[x_column],
            granularity=granularity,
        )

        cumulative_data[group_key] = cumulative_df

    return cumulative_data


def _plot_cumulative_line(
    ax: Axes,
    cumulative_df: nw.DataFrame,
    orientation: Literal["vertical", "horizontal"],
    **kwargs: Any,
) -> Line2D:
    """Plot cumulative distribution line on the given axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    cumulative_df : nw.DataFrame
        Cumulative distribution data with 'position' and 'cumulative_probability'
    orientation : {'vertical', 'horizontal'}
        Line orientation
    **kwargs : dict
        Additional arguments passed to plot()

    Returns
    -------
    Line2D
        The plotted line object
    """
    # Extract cumulative data
    x_coords = cumulative_df["position"].to_list()
    y_coords = cumulative_df["cumulative_probability"].to_list()

    # Plot line based on orientation
    if orientation == "vertical":
        lines = ax.plot(x_coords, y_coords, **kwargs)
    else:  # horizontal
        lines = ax.plot(y_coords, x_coords, **kwargs)

    return lines[0]


def _setup_axes_labels(
    ax: Axes,
    x_column: str,
    orientation: Literal["vertical", "horizontal"],
    xlabel: Optional[str],
    ylabel: Optional[str],
) -> None:
    """Setup axis labels based on plot configuration.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to configure
    x_column : str
        Name of the data column
    orientation : {'vertical', 'horizontal'}
        Line orientation
    xlabel : str or None
        Custom x-axis label
    ylabel : str or None
        Custom y-axis label
    """
    # Cumulative probability label
    cumulative_label = "Cumulative Probability"

    # Apply labels based on orientation
    if orientation == "vertical":
        ax.set_xlabel(xlabel if xlabel is not None else x_column)
        ax.set_ylabel(ylabel if ylabel is not None else cumulative_label)
    else:
        ax.set_ylabel(ylabel if ylabel is not None else x_column)
        ax.set_xlabel(xlabel if xlabel is not None else cumulative_label)


def cumulative(
    data: Optional[IntoDataFrame] = None,
    *,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    hue: Optional[str] = None,
    ax: Optional[Axes] = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    granularity: Optional[GranularityT] = "best",
    alpha: Optional[float] = None,
    linewidth: Optional[float] = None,
    linestyle: Optional[str] = None,
    color: Optional[Union[str, list[str]]] = None,
    palette: Optional[Union[str, list[str]]] = None,
    marker: Optional[str] = None,
    markersize: Optional[float] = None,
    markevery: Optional[Union[int, tuple]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend: bool = True,
    **kwargs: Any,
) -> Line2D | list[Line2D] | None:
    """Create a cumulative distribution plot using Khisto's optimal binning algorithm.

    This function provides a matplotlib-compatible interface for creating
    cumulative distribution function (CDF) plots with automatic optimal bin
    selection using the Khisto algorithm. The CDF shows the probability that
    a random variable takes a value less than or equal to x.

    The function integrates seamlessly with matplotlib's ecosystem:
    - Accepts an `ax` parameter for subplot integration
    - Returns matplotlib Line2D objects for further customization
    - Supports standard matplotlib styling parameters
    - Works with matplotlib's figure/axes workflow

    Parameters
    ----------
    data : IntoDataFrame, optional
        A DataFrame-like object (pandas, polars, etc.) containing the data.
        If None, `x` must be an array-like object.
    x : str or ArrayT or IntoSeries, optional
        Either a column name in `data`, or an array/Series for the values
        to compute the CDF. Supports NumPy arrays, pandas/polars Series, lists,
        tuples, and any array supporting the Python Array API standard.
    hue : str, optional
        Column name in `data` for grouping. Creates separate CDF lines
        for each unique value with different colors.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes (plt.gca()).
        This allows integration into subplot grids and custom figure layouts.
    orientation : {'vertical', 'horizontal'}, default 'vertical'
        Orientation of the cumulative distribution plot.
        - 'vertical': x-axis shows values, y-axis shows cumulative probability
        - 'horizontal': y-axis shows values, x-axis shows cumulative probability
    granularity : int or 'best' or None, default 'best'
        Granularity level to use for binning.
        - 'best': Uses the optimal granularity level (default)
        - int: Uses the specified granularity level
        - None: Uses all available granularities (returns list of lines)
    alpha : float, optional
        Transparency level for lines (0.0 to 1.0). Lower values create
        transparency, useful for overlapping lines.
    linewidth : float, optional
        Width of the line in points.
    linestyle : str, optional
        Style of the line. Examples: '-', '--', '-.', ':', 'solid', 'dashed',
        'dashdot', 'dotted'
    color : str or list of str, optional
        Color(s) for the lines. If `hue` is used, can be a list matching
        the number of hue categories. Any matplotlib color specification.
    palette : str or list of str, optional
        Color palette name (e.g., 'viridis', 'Set1') or list of colors
        to use for different hue groups. Ignored if `hue` is None.
    marker : str, optional
        Marker style for data points. Examples: 'o', 's', '^', 'v', 'D', '*'
    markersize : float, optional
        Size of markers in points.
    markevery : int or tuple, optional
        Draw markers at every N points, or at specific indices/ranges.
        Examples: 5 (every 5th point), (0.1, 0.2) (10-20% of points)
    xlabel : str, optional
        Label for x-axis. If None, uses the column name.
    ylabel : str, optional
        Label for y-axis. If None, uses 'Cumulative Probability'.
    title : str, optional
        Title for the plot.
    legend : bool, default True
        Whether to show legend when using `hue` grouping.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib's `plot()` function.
        Examples: 'label', 'zorder', 'drawstyle'

    Returns
    -------
    Line2D or list[Line2D]
        If `hue` is None: returns a single Line2D object.
        If `hue` is used: returns a list of Line2D objects, one for each group.

        Line2D objects can be used for further customization of the plot.

    Examples
    --------
    Create a simple cumulative distribution plot from a NumPy array:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from khisto.plot.matplotlib import cumulative
    >>> data = np.random.normal(0, 1, 1000)
    >>> cumulative(x=data)
    >>> plt.show()

    Create a cumulative plot from a DataFrame with grouping:

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'value': np.random.normal(0, 1, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000)
    ... })
    >>> cumulative(data=df, x='value', hue='category')
    >>> plt.show()

    Create a horizontal cumulative plot on a specific axes:

    >>> fig, ax = plt.subplots()
    >>> cumulative(x=data, ax=ax, orientation='horizontal', title='CDF Plot')
    >>> plt.show()

    Customize styling with matplotlib parameters:

    >>> cumulative(
    ...     x=data,
    ...     alpha=0.7,
    ...     linewidth=2,
    ...     linestyle='--',
    ...     color='blue',
    ...     marker='o',
    ...     markevery=20
    ... )
    >>> plt.show()

    Use in a subplot grid:

    >>> fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    >>> for i, ax in enumerate(axes.flat):
    ...     data = np.random.normal(i, 1, 500)
    ...     cumulative(x=data, ax=ax, title=f'CDF {i}')
    >>> plt.tight_layout()
    >>> plt.show()

    Compare multiple distributions:

    >>> fig, ax = plt.subplots()
    >>> for i, loc in enumerate([0, 1, 2]):
    ...     data = np.random.normal(loc, 1, 500)
    ...     cumulative(x=data, ax=ax, label=f'μ={loc}', alpha=0.7)
    >>> ax.legend()
    >>> plt.show()

    See Also
    --------
    khisto.array.ecdf_values_table : Get cumulative distribution as DataFrame
    khisto.plot.matplotlib.histogram : Create histogram plots with optimal binning
    matplotlib.pyplot.plot : Standard matplotlib line plotting function

    Notes
    -----
    This function uses the Khisto algorithm for automatic optimal bin selection
    when computing the cumulative distribution. The algorithm analyzes the data
    distribution to determine the best binning strategy.

    The cumulative distribution function (CDF) shows the probability that a
    random variable X takes a value less than or equal to x:

        F(x) = P(X ≤ x)

    The CDF always ranges from 0 to 1, starting at 0 for the minimum value
    and reaching 1 at the maximum value. The plot is created as a step function
    where each step corresponds to a bin edge from Khisto's optimal binning.

    Key features:

    - **Automatic bin selection**: No need to specify number of bins or bin width
    - **Smooth CDF**: The cumulative distribution is computed from optimally
      selected bins, providing an accurate representation
    - **Step function visualization**: Each step shows the cumulative probability
      at bin boundaries
    """
    # Get or create axes
    if ax is None:
        ax = plt.gca()

    # Process input data
    df, x_column = _process_input_data(data, x)

    # Compute cumulative data for each group
    cumulative_data = _compute_cumulative_data(df, x_column, hue, granularity)

    # Setup axis labels (do this even if no data)
    _setup_axes_labels(ax, x_column, orientation, xlabel, ylabel)

    # Add title if provided
    if title is not None:
        ax.set_title(title)

    # Set y-axis range for vertical orientation (cumulative probability)
    if orientation == "vertical":
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.set_xlim(-0.05, 1.05)

    # If all groups are empty, return None
    if not cumulative_data or all(len(cdf) == 0 for cdf in cumulative_data.values()):
        return None

    # Determine colors
    if palette is not None and hue is not None:
        # Use palette for colors
        if isinstance(palette, str):
            import matplotlib.cm as cm

            cmap = cm.get_cmap(palette)
            n_colors = len(cumulative_data)
            colors = [cmap(i / n_colors) for i in range(n_colors)]
        else:
            colors = list(palette)
    elif color is not None:
        if isinstance(color, (list, tuple)):
            colors = list(color)
        else:
            colors = [color] * len(cumulative_data)
    else:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    # Build kwargs for line plot
    line_kwargs = kwargs.copy()
    if alpha is not None:
        line_kwargs["alpha"] = alpha
    if linewidth is not None:
        line_kwargs["linewidth"] = linewidth
    if linestyle is not None:
        line_kwargs["linestyle"] = linestyle
    if marker is not None:
        line_kwargs["marker"] = marker
    if markersize is not None:
        line_kwargs["markersize"] = markersize
    if markevery is not None:
        line_kwargs["markevery"] = markevery

    # Plot cumulative lines for each group
    lines = []
    for idx, (group_key, cumulative_df) in enumerate(cumulative_data.items()):
        if len(cumulative_df) == 0:
            continue
        group_kwargs = line_kwargs.copy()
        group_kwargs["color"] = colors[idx % len(colors)]
        if hue is not None and group_key is not None:
            group_kwargs["label"] = str(group_key)
        line = _plot_cumulative_line(ax, cumulative_df, orientation, **group_kwargs)
        lines.append(line)

    # Add legend if using hue and legend is True
    if hue is not None and legend and len(lines) > 1:
        ax.legend()

    # Return lines
    if hue is not None:
        return lines
    if len(lines) == 1:
        return lines[0]
    return lines if lines else None

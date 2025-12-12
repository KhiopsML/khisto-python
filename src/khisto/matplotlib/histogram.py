"""Matplotlib histogram visualization with optimal binning.

This module provides a matplotlib-compatible histogram function that uses
Khisto's optimal binning algorithm for automatic bin selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union, Any

import pyarrow as pa
import narwhals as nw

from khisto.array import histogram_table
from khisto.utils._compat._optional import import_optional_dependency, Extras

import_optional_dependency("matplotlib", extra=Extras.MATPLOTLIB, errors="raise")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

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
                "Either pass x as a column name, or use histogram(x=array) without the data parameter."
            )
        x_column = x

    return df, x_column


def _compute_histogram_data(
    df: nw.DataFrame[Any] | nw.LazyFrame[Any],
    x_column: str,
    hue: Optional[str],
    granularity: Optional[GranularityT],
) -> dict[Any, nw.DataFrame[Any]]:
    """Compute histogram data for each group.

    Parameters
    ----------
    df : nw.DataFrame
        Input DataFrame
    x_column : str
        Name of column containing values to histogram
    hue : str or None
        Column name for grouping/coloring
    granularity : int, 'best', or None
        Granularity level for binning

    Returns
    -------
    dict[Any, nw.DataFrame]
        Dictionary mapping group keys to their histogram DataFrames
    """
    # If no grouping, treat as single group
    if hue is None:
        groups = {None: df}
    else:
        # Group by hue column
        unique_values = df[hue].unique(maintain_order=True).to_list()
        groups = {val: df.filter(nw.col(hue) == val) for val in unique_values}

    # Compute histogram for each group
    histo_data = {}
    for group_key, group_df in groups.items():
        # Use histogram_table which returns histogram data
        histo_df = histogram_table(
            group_df[x_column],
            granularity=granularity,
        )

        histo_data[group_key] = histo_df

    return histo_data


def _plot_histogram_bars(
    ax: Axes,
    histo_df: nw.DataFrame,
    orientation: Literal["vertical", "horizontal"],
    **kwargs: Any,
) -> BarContainer:
    """Plot histogram bars on the given axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to plot on
    histo_df : nw.DataFrame
        Histogram data
    orientation : {'vertical', 'horizontal'}
        Bar orientation
    **kwargs : dict
        Additional arguments passed to bar()

    Returns
    -------
    BarContainer
        Container with bar patches
    """
    # Extract histogram data
    lower_bounds = histo_df["lower_bound"].to_list()
    upper_bounds = histo_df["upper_bound"].to_list()
    widths = histo_df["length"].to_list()
    heights = histo_df["density"].to_list()

    # Compute bin centers for positioning
    centers = [(lower + upper) / 2 for lower, upper in zip(lower_bounds, upper_bounds)]

    # Plot bars based on orientation
    if orientation == "vertical":
        container = ax.bar(
            centers,
            heights,
            width=widths,
            align="center",
            **kwargs,
        )
    else:  # horizontal
        container = ax.barh(
            centers,
            heights,
            height=widths,
            align="center",
            **kwargs,
        )

    return container


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
        Bar orientation
    xlabel : str or None
        Custom x-axis label
    ylabel : str or None
        Custom y-axis label
    """
    # Density label
    density_label = "Density"

    # Apply labels based on orientation
    if orientation == "vertical":
        ax.set_xlabel(xlabel if xlabel is not None else x_column)
        ax.set_ylabel(ylabel if ylabel is not None else density_label)
    else:
        ax.set_ylabel(ylabel if ylabel is not None else x_column)
        ax.set_xlabel(xlabel if xlabel is not None else density_label)


def histogram(
    data: Optional[IntoDataFrame] = None,
    *,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    hue: Optional[str] = None,
    ax: Optional[Axes] = None,
    orientation: Literal["vertical", "horizontal"] = "vertical",
    granularity: Optional[GranularityT] = "best",
    alpha: Optional[float] = None,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    color: Optional[Union[str, list[str]]] = None,
    palette: Optional[Union[str, list[str]]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    legend: bool = True,
    **kwargs: Any,
) -> BarContainer | list[BarContainer]:
    """Create a histogram using Khisto's optimal binning algorithm.

    This function provides a matplotlib-compatible interface for creating
    histograms with automatic optimal bin selection using the Khisto algorithm.
    Unlike standard histograms that require manual bin specification, Khisto
    automatically determines the optimal number of bins and their boundaries
    to best represent the underlying data distribution.

    The function integrates seamlessly with matplotlib's ecosystem:
    - Accepts an `ax` parameter for subplot integration
    - Returns matplotlib container objects for further customization
    - Supports standard matplotlib styling parameters
    - Works with matplotlib's figure/axes workflow

    Parameters
    ----------
    data : IntoDataFrame, optional
        A DataFrame-like object (pandas, polars, etc.) containing the data.
        If None, `x` must be an array-like object.
    x : str or ArrayT or IntoSeries, optional
        Either a column name in `data`, or an array/Series for the values
        to bin. Supports NumPy arrays, pandas/polars Series, lists, tuples,
        and any array supporting the Python Array API standard.
    hue : str, optional
        Column name in `data` for grouping. Creates separate histogram bars
        for each unique value with different colors.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot on. If None, uses current axes (plt.gca()).
        This allows integration into subplot grids and custom figure layouts.
    orientation : {'vertical', 'horizontal'}, default 'vertical'
        Orientation of histogram bars.
        - 'vertical': Vertical bars with x-axis showing values
        - 'horizontal': Horizontal bars with y-axis showing values
    granularity : int or 'best' or None, default 'best'
        Granularity level to use for histogram binning.
        - 'best': Uses the optimal granularity level (default)
        - int: Uses the specified granularity level
        - None: Uses all available granularities (returns list of containers)
    alpha : float, optional
        Transparency level for bars (0.0 to 1.0). Lower values create
        transparency, useful for overlapping histograms.
    edgecolor : str, optional
        Color of bar edges. Any matplotlib color specification.
        Examples: 'black', '#000000', (0, 0, 0)
    linewidth : float, optional
        Width of bar edges in points.
    color : str or list of str, optional
        Color(s) for the bars. If `hue` is used, can be a list matching
        the number of hue categories. Any matplotlib color specification.
    palette : str or list of str, optional
        Color palette name (e.g., 'viridis', 'Set1') or list of colors
        to use for different hue groups. Ignored if `hue` is None.
    xlabel : str, optional
        Label for x-axis. If None, uses the column name.
    ylabel : str, optional
        Label for y-axis. If None, uses 'Density'.
    title : str, optional
        Title for the plot.
    legend : bool, default True
        Whether to show legend when using `hue` grouping.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib's `bar()` or `barh()`
        functions. Examples: 'label', 'zorder', 'hatch'

    Returns
    -------
    BarContainer or list[BarContainer]
        If `hue` is None: returns a single BarContainer.
        If `hue` is used: returns a list of BarContainers, one for each group.

        BarContainer objects can be used for further customization of the plot.

    Examples
    --------
    Create a simple histogram from a NumPy array:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from khisto.plot.matplotlib import histogram
    >>> data = np.random.normal(0, 1, 1000)
    >>> histogram(x=data)
    >>> plt.show()

    Create a histogram from a DataFrame with grouping:

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'value': np.random.normal(0, 1, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000)
    ... })
    >>> histogram(data=df, x='value', hue='category')
    >>> plt.show()

    Create a horizontal histogram on a specific axes:

    >>> fig, ax = plt.subplots()
    >>> histogram(x=data, ax=ax, orientation='horizontal', title='My Histogram')
    >>> plt.show()

    Customize styling with matplotlib parameters:

    >>> fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    >>> for i, ax in enumerate(axes.flat):
    ...     data = np.random.normal(i, 1, 500)
    ...     histogram(x=data, ax=ax, title=f'Distribution {i}')
    >>> plt.tight_layout()
    >>> plt.show()

    See Also
    --------
    khisto.array.histogram_table : Get histogram information as DataFrame
    khisto.plot.matplotlib.cumulative : Create cumulative distribution plots
    matplotlib.pyplot.hist : Standard matplotlib histogram function

    Notes
    -----
    This function uses the Khisto algorithm for automatic optimal bin selection.
    The algorithm analyzes the data distribution to determine the best binning
    strategy that maximizes information while minimizing complexity.

    Key differences from matplotlib.pyplot.hist:
    - Automatic bin selection (no need to specify bins parameter)
    - Variable bin widths optimized for the data distribution
    - Density-based by default (y-axis is probability density)
    - Returns BarContainer objects for consistency with matplotlib

    The histogram bars use Khisto's optimal bins with potentially variable widths.
    Each bar's height represents the probability density for that bin, computed as
    the proportion of data points divided by the bin width. This ensures the total
    area of all bars equals 1.
    """
    # Get or create axes
    if ax is None:
        ax = plt.gca()

    # Process input data
    df, x_column = _process_input_data(data, x)

    # Compute histogram data for each group
    histo_data = _compute_histogram_data(df, x_column, hue, granularity)

    # Determine colors
    if palette is not None and hue is not None:
        # Use palette for colors
        if isinstance(palette, str):
            # Named colormap
            import matplotlib.cm as cm

            cmap = cm.get_cmap(palette)
            n_colors = len(histo_data)
            colors = [cmap(i / n_colors) for i in range(n_colors)]
        else:
            colors = list(palette)
    elif color is not None:
        if isinstance(color, (list, tuple)):
            colors = list(color)
        else:
            colors = [color] * len(histo_data)
    else:
        # Use matplotlib's default color cycle
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    # Build kwargs for bar plot
    bar_kwargs = kwargs.copy()
    if alpha is not None:
        bar_kwargs["alpha"] = alpha
    if edgecolor is not None:
        bar_kwargs["edgecolor"] = edgecolor
    if linewidth is not None:
        bar_kwargs["linewidth"] = linewidth

    # Plot histogram bars for each group
    containers = []
    for idx, (group_key, histo_df) in enumerate(histo_data.items()):
        # Set color for this group
        group_kwargs = bar_kwargs.copy()
        group_kwargs["color"] = colors[idx % len(colors)]

        # Set label for legend
        if hue is not None and group_key is not None:
            group_kwargs["label"] = str(group_key)

        # Plot bars
        container = _plot_histogram_bars(ax, histo_df, orientation, **group_kwargs)
        containers.append(container)

    # Setup axis labels
    _setup_axes_labels(ax, x_column, orientation, xlabel, ylabel)

    # Add title if provided
    if title is not None:
        ax.set_title(title)

    # Add legend if using hue and legend is True
    if hue is not None and legend and len(containers) > 1:
        ax.legend()

    # Return containers
    if len(containers) == 1:
        return containers[0]
    return containers

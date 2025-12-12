"""Plotly cumulative distribution visualization with optimal binning.

This module provides a Plotly Express-compatible cumulative distribution function
that uses Khisto's optimal binning algorithm for automatic bin selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import pyarrow as pa
import narwhals as nw

from khisto.array import ecdf_values_table
from khisto.utils._compat._optional import import_optional_dependency, Extras

import_optional_dependency("plotly", extra=Extras.PLOTLY, errors="raise")
from plotly.express._core import (
    make_figure,
    build_dataframe,
    infer_config,
    get_groups_and_orders,
    one_group,
    apply_default_cascade,
)
import plotly.graph_objects as go

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoDataFrame, IntoSeries


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_cumulative_for_groups(
    groups: dict,
    x_column_name: str,
    granularity: Optional[GranularityT],
    grouped_mappings: list,
) -> list[nw.DataFrame]:
    """Compute cumulative distribution data for each group in the data.

    Parameters
    ----------
    groups : dict
        Dictionary mapping group identifiers to DataFrames
    x_column_name : str
        Name of the column containing values
    granularity : int, 'best', or None
        Granularity level for histogram computation
    grouped_mappings : list
        List of grouping mappings from Plotly Express

    Returns
    -------
    list[nw.DataFrame]
        List of DataFrames containing cumulative distribution data for each group
    """
    cumulative_df_list: list[nw.DataFrame] = []

    for group_key, group_df in groups.items():
        # Determine the actual column name in this group's DataFrame
        value_column_name = x_column_name if x_column_name in group_df.columns else "x"

        # Compute cumulative distribution directly
        cdf_df = ecdf_values_table(group_df[value_column_name], granularity=granularity)

        # Create line plot data points from CDF
        cumulative_points = []

        for g in cdf_df["granularity"].unique(maintain_order=True):
            granularity_df = cdf_df.filter(nw.col("granularity") == g)

            # Rename position to x and cumulative_probability to y for plotting
            points_df = granularity_df.rename(
                {"position": "x", "cumulative_probability": "y"}
            )

            # Add offsetgroup for proper trace grouping
            points_df = points_df.with_columns(
                offsetgroup=nw.lit(", ".join(str(v) for v in group_key if v))
            )

            # Add group identifiers
            points_df = points_df.with_columns(
                **{
                    mapping.grouper: nw.lit(value)
                    for mapping, value in zip(grouped_mappings, group_key)
                    if mapping is not None and mapping.grouper is not None
                }
            )

            cumulative_points.append(points_df)

        # Combine all granularity levels for this group
        if cumulative_points:
            cumulative_df_list.append(nw.concat(cumulative_points))

    return cumulative_df_list


def _fill_missing_granularities(
    cumulative_df_list: list[nw.DataFrame],
) -> list[nw.DataFrame]:
    """Fill missing granularity levels across all groups.

    When different groups have different maximum granularities, this function
    duplicates the highest granularity level to fill in missing levels. This
    ensures all groups have the same granularity levels for synchronized
    animation.

    Parameters
    ----------
    cumulative_df_list : list[nw.DataFrame]
        List of cumulative DataFrames, possibly with different max granularities

    Returns
    -------
    list[nw.DataFrame]
        List of cumulative DataFrames with consistent granularity levels
    """
    if not cumulative_df_list:
        return cumulative_df_list

    # Find the maximum granularity across all groups
    max_granularity = max(df["granularity"].max() for df in cumulative_df_list)

    # Fill missing granularity levels for each group
    filled_list = []
    for cumulative_df in cumulative_df_list:
        actual_max_granularity = cumulative_df["granularity"].max()

        if actual_max_granularity < max_granularity:
            # Duplicate the highest granularity data for missing levels
            duplicates = [
                cumulative_df.filter(
                    nw.col("granularity") == actual_max_granularity
                ).with_columns(
                    granularity=nw.lit(g),
                    is_best=nw.lit(False),
                )
                for g in range(actual_max_granularity + 1, max_granularity + 1)
            ]
            filled_list.append(nw.concat([cumulative_df, *duplicates]))
        else:
            filled_list.append(cumulative_df)

    return filled_list


def _create_empty_cumulative_dataframe() -> nw.DataFrame:
    """Create an empty cumulative DataFrame with correct schema.

    Returns
    -------
    nw.DataFrame
        Empty DataFrame with cumulative columns
    """
    return nw.from_dict(
        {
            "x": pa.array([]),
            "y": pa.array([]),
            "offsetgroup": pa.array([]),
            "granularity": pa.array([]),
        },
        backend="pyarrow",
    )


def _combine_cumulative_dataframes(
    cumulative_df_list: list[nw.DataFrame],
) -> nw.DataFrame:
    """Combine cumulative DataFrames from all groups into a single DataFrame.

    Parameters
    ----------
    cumulative_df_list : list[nw.DataFrame]
        List of cumulative DataFrames for each group

    Returns
    -------
    nw.DataFrame
        Combined and sorted cumulative DataFrame
    """
    if not cumulative_df_list:
        return _create_empty_cumulative_dataframe()

    combined_df = nw.concat(cumulative_df_list)
    return combined_df.sort(["granularity", "offsetgroup", "x"])


def _determine_best_granularity(
    combined_cumulative_df: nw.DataFrame,
    granularity: Optional[GranularityT],
) -> int:
    """Determine the best granularity level to display initially.

    Parameters
    ----------
    combined_cumulative_df : nw.DataFrame
        Combined cumulative DataFrame with all granularities
    granularity : int, 'best', or None
        User-specified granularity preference

    Returns
    -------
    int
        The granularity level to display initially
    """
    if isinstance(granularity, int):
        return granularity

    # Handle empty dataframe
    if len(combined_cumulative_df) == 0:
        return 0

    if granularity == "best":
        max_val = combined_cumulative_df["granularity"].max()
        return int(max_val) if max_val is not None else 0
    else:  # granularity is None
        best_df = combined_cumulative_df.filter(nw.col("is_best"))
        if len(best_df) > 0:
            max_val = best_df["granularity"].max()
            return int(max_val) if max_val is not None else 0
        else:
            # Fallback to overall max if no is_best found
            max_val = combined_cumulative_df["granularity"].max()
            return int(max_val) if max_val is not None else 0


def _configure_animation_axes(
    fig: go.Figure,
    combined_cumulative_df: nw.DataFrame,
    orientation: str,
    best_granularity: int,
) -> None:
    """Configure axis ranges for animated cumulative plots.

    Sets fixed axis ranges based on the overall data range to prevent
    axes from jumping during animation.

    Parameters
    ----------
    fig : go.Figure
        The Plotly Figure to configure
    combined_cumulative_df : nw.DataFrame
        Combined cumulative DataFrame with all granularities
    orientation : str
        'v' for vertical or 'h' for horizontal
    best_granularity : int
        The best granularity level for initial display
    """
    # Calculate data range with margin
    min_x = combined_cumulative_df["x"].min()
    max_x = combined_cumulative_df["x"].max()
    axis_margin = 0.05 * (max_x - min_x)

    # Set appropriate axis ranges based on orientation
    if orientation == "v":
        fig.update_xaxes(
            range=(min_x - axis_margin, max_x + axis_margin),
        )
        fig.update_yaxes(range=(-0.05, 1.05))
    else:
        fig.update_yaxes(
            range=(min_x - axis_margin, max_x + axis_margin),
        )
        fig.update_xaxes(range=(-0.05, 1.05))


def cumulative(
    data_frame: Optional[IntoDataFrame] = None,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    y: Optional[Union[str, ArrayT, IntoSeries]] = None,
    color: Optional[str] = None,
    line_dash: Optional[str] = None,
    symbol: Optional[str] = None,
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    facet_col_wrap: Optional[int] = None,
    facet_row_spacing: Optional[float] = None,
    facet_col_spacing: Optional[float] = None,
    hover_name: Optional[str] = None,
    hover_data: Optional[Union[list[str], dict]] = None,
    animation_frame: Optional[str] = None,
    animation_group: Optional[str] = None,
    category_orders: Optional[dict] = None,
    labels: Optional[dict] = None,
    color_discrete_sequence: Optional[list[str]] = None,
    color_discrete_map: Optional[dict] = None,
    line_dash_sequence: Optional[list[str]] = None,
    line_dash_map: Optional[dict] = None,
    symbol_sequence: Optional[list[str]] = None,
    symbol_map: Optional[dict] = None,
    markers: bool = False,
    orientation: Optional[Literal["v", "h"]] = "v",
    log_x: bool = False,
    log_y: bool = False,
    range_x: Optional[list] = None,
    range_y: Optional[list] = None,
    line_shape: Optional[
        Literal["linear", "hv", "vh", "hvh", "vhv", "spline"]
    ] = "linear",
    render_mode: Literal["auto", "svg", "webgl"] = "auto",
    granularity: Optional[GranularityT] = "best",
    text: Optional[str] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    template: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Create a cumulative distribution plot using Khisto's optimal binning algorithm.

    This function provides a Plotly Express-compatible interface for creating
    cumulative distribution function (CDF) plots with automatic optimal bin
    selection using the Khisto algorithm. The CDF shows the probability that
    a random variable takes a value less than or equal to x.

    The function accepts data in multiple formats:

    - Direct arrays: NumPy arrays, lists, tuples
    - Pandas/Polars Series or DataFrames
    - Any array supporting the Python Array API standard

    When using DataFrames, additional Plotly Express features are available
    including color grouping, faceting, animations, and line styling.

    Parameters
    ----------
    data_frame : IntoDataFrame, optional
        A DataFrame-like object (pandas, polars, etc.) or a Narwhals DataFrame.
        Either `data_frame` or direct array input via `x` must be provided.
    x : str or ArrayT or IntoSeries, optional
        Either a column name in `data_frame`, or an array/Series for the values
        to compute the CDF. Supports NumPy arrays, pandas/polars Series, lists,
        tuples, and any array supporting the Python Array API standard.
    y : str or ArrayT or IntoSeries, optional
        Not used for cumulative plots. Included for Plotly Express compatibility.
    color : str, optional
        Column name in `data_frame` for color encoding. Creates separate CDF
        lines for each unique value in this column.
    line_dash : str, optional
        Column name in `data_frame` for line dash encoding. Uses different
        line dash patterns for different values.
    symbol : str, optional
        Column name in `data_frame` for symbol encoding when markers=True.
        Uses different marker symbols for different values.
    facet_row : str, optional
        Column name in `data_frame` for faceting into subplot rows. Creates a
        separate row of subplots for each unique value.
    facet_col : str, optional
        Column name in `data_frame` for faceting into subplot columns. Creates a
        separate column of subplots for each unique value.
    facet_col_wrap : int, optional
        Maximum number of facet columns. Wraps to new rows if exceeded. Useful
        for creating grid layouts with many facets.
    facet_row_spacing : float, optional
        Spacing between facet rows as a fraction (0 to 1). Default spacing is
        applied if not specified.
    facet_col_spacing : float, optional
        Spacing between facet columns as a fraction (0 to 1). Default spacing is
        applied if not specified.
    hover_name : str, optional
        Column name in `data_frame` to display as bold text in hover tooltips.
    hover_data : list of str or dict, optional
        Additional columns to include in hover tooltips. Can be a list of column
        names or a dict mapping column names to formatting specifications
        (e.g., {'price': ':.2f'} for 2 decimal places).
    animation_frame : str, optional
        Column name in `data_frame` for animation frame grouping. Creates an
        animated CDF plot with a timeline slider.
    animation_group : str, optional
        Column name in `data_frame` for matching objects across animation frames.
        Ensures smooth transitions between frames.
    category_orders : dict, optional
        Dict mapping categorical column names to ordered lists of values. Controls
        the order of categorical values in legends, facets, and animations.
        Example: {'day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']}
    labels : dict, optional
        Dict mapping column names to custom axis/legend labels. Used to provide
        human-readable names for display.
        Example: {'x': 'Temperature (°C)', 'y': 'Cumulative Probability'}
    color_discrete_sequence : list of str, optional
        List of CSS color strings to cycle through for discrete color values.
        Example: ['#FF0000', '#00FF00', '#0000FF']
    color_discrete_map : dict, optional
        Dict mapping specific discrete color values to CSS colors. Provides
        complete control over which values get which colors.
        Example: {'A': 'red', 'B': 'blue', 'C': 'green'}
    line_dash_sequence : list of str, optional
        List of line dash patterns to cycle through. Available patterns include
        'solid', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot'.
    line_dash_map : dict, optional
        Dict mapping line dash values to specific patterns. Provides complete
        control over dash pattern assignment.
    symbol_sequence : list of str, optional
        List of marker symbols to cycle through when markers=True. Available
        symbols include 'circle', 'square', 'diamond', 'cross', 'x', etc.
    symbol_map : dict, optional
        Dict mapping symbol values to specific marker symbols. Provides complete
        control over symbol assignment.
    markers : bool, default False
        If True, show markers at data points. Creates a line plot with markers.
    orientation : {'v', 'h'}, default 'v'
        Orientation of the plot.
        - 'v': Vertical (default) with x-axis showing values, y-axis showing probability
        - 'h': Horizontal with y-axis showing values, x-axis showing probability
    log_x : bool, default False
        If True, use logarithmic scale for x-axis. Useful for data spanning
        multiple orders of magnitude.
    log_y : bool, default False
        If True, use logarithmic scale for y-axis (cumulative probability axis).
    range_x : list, optional
        Range for x-axis as [min, max]. Restricts the visible range of the
        plot. Example: [-3, 3]
    range_y : list, optional
        Range for y-axis as [min, max]. Restricts the visible range of the
        probability axis. Example: [0, 1]
    line_shape : {'linear', 'hv', 'vh', 'hvh', 'vhv', 'spline'}, default 'linear'
        Shape of the line connecting points.
        - 'linear': Straight line segments (default)
        - 'hv': Horizontal then vertical (step function)
        - 'vh': Vertical then horizontal
        - 'hvh': Horizontal, vertical, horizontal
        - 'vhv': Vertical, horizontal, vertical
        - 'spline': Smooth curve
    render_mode : {'auto', 'svg', 'webgl'}, default 'auto'
        Rendering mode for the plot. 'webgl' is faster for large datasets.
    granularity : int or 'best' or None, default 'best'
        Granularity level to use for binning.
        - 'best': Uses the optimal granularity level (default)
        - int: Uses the specified granularity level
        - None: Creates an interactive slider to explore all granularity levels
        When None and multiple groups exist (via color, facets, etc.), the slider
        synchronizes across all groups to show the same granularity level.
    text : str, optional
        Column name in `data_frame` to use for text labels on points.
    title : str, optional
        Figure title displayed at the top of the plot.
    subtitle : str, optional
        Figure subtitle displayed below the title. Note: May not be supported
        in all Plotly versions.
    template : str, optional
        Plotly template name controlling overall styling and theme.
        Examples: 'plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn',
        'simple_white', 'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none'
    width : int, optional
        Figure width in pixels. Example: 800
    height : int, optional
        Figure height in pixels. Example: 600

    Returns
    -------
    go.Figure
        A Plotly Figure object with the cumulative distribution visualization.
        The figure can be displayed using `fig.show()`, saved using
        `fig.write_html()` or `fig.write_image()`, or further customized using
        Plotly's API.

    Examples
    --------
    Create a simple CDF plot from a NumPy array:

    >>> import numpy as np
    >>> from khisto.plot.plotly import cumulative
    >>> data = np.random.normal(0, 1, 1000)
    >>> fig = cumulative(x=data)
    >>> fig.show()

    Create a CDF plot from a DataFrame with color grouping:

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'value': np.random.normal(0, 1, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000)
    ... })
    >>> fig = cumulative(df, x='value', color='category')
    >>> fig.show()

    Create a horizontal CDF with custom styling:

    >>> fig = cumulative(
    ...     x=data,
    ...     orientation='h',
    ...     title='Cumulative Distribution',
    ...     template='plotly_white',
    ...     line_shape='linear'
    ... )
    >>> fig.show()

    Create a CDF with markers and custom line shape:

    >>> fig = cumulative(x=data, markers=True, line_shape='linear')
    >>> fig.show()

    Compare multiple distributions:

    >>> fig = cumulative(
    ...     df,
    ...     x='value',
    ...     color='category',
    ...     line_dash='category',
    ...     title='Comparing Distributions'
    ... )
    >>> fig.show()

    Create an interactive granularity explorer:

    >>> fig = cumulative(x=data, granularity=None)
    >>> fig.show()

    See Also
    --------
    khisto.array.ecdf : Compute callable ECDF object
    khisto.array.ecdf_values_table : Get full CDF information as DataFrame
    khisto.plot.plotly.histogram : Create histogram plots with optimal binning

    Notes
    -----
    This function uses the Khisto algorithm for automatic optimal bin selection
    when computing the cumulative distribution. The Khisto algorithm analyzes
    the data distribution to determine the best binning strategy.

    Key features:

    - **Automatic bin selection**: No need to specify number of bins or bin width
    - **Smooth CDF**: The cumulative distribution is computed from optimally
      selected bins, providing a smooth and accurate representation
    - **Multi-granularity**: Khisto can analyze data at multiple scales to find
      the optimal representation

    The cumulative distribution function (CDF) shows the probability that a
    random variable X takes a value less than or equal to x:

        F(x) = P(X ≤ x)

    The CDF always ranges from 0 to 1, starting at 0 for the minimum value
    and reaching 1 at the maximum value.

    When using color grouping, faceting, or animations, Khisto computes optimal
    bins independently for each group, ensuring the best representation for each
    subset of data.
    """
    # Step 1: Prepare arguments and build dataframe structure
    args = locals().copy()
    granularity_param = args.pop("granularity")

    apply_default_cascade(args)
    args = build_dataframe(args, go.Scatter)

    # Step 2: Infer plot configuration and group data
    trace_specs, grouped_mappings, sizeref, show_colorbar = infer_config(
        args.copy(), go.Scatter, {}, {}
    )
    grouper = [x.grouper or one_group for x in grouped_mappings] or [one_group]
    groups, orders = get_groups_and_orders(args.copy(), grouper)

    # Step 3: Compute cumulative distribution data for each group
    x_column_name = "_value" if not isinstance(x, str) else args["x"]
    cumulative_df_list = _compute_cumulative_for_groups(
        groups, x_column_name, granularity_param, grouped_mappings
    )

    # Step 4: Ensure consistent granularity levels across groups (for animation)
    if granularity_param is None:
        cumulative_df_list = _fill_missing_granularities(cumulative_df_list)

    # Step 5: Combine all cumulative data into a single DataFrame
    combined_cumulative_df = _combine_cumulative_dataframes(cumulative_df_list)

    # Step 6: Prepare data for Plotly figure creation
    args["data_frame"] = nw.to_native(combined_cumulative_df)
    args["x"] = "x" if orientation == "v" else "y"
    args["y"] = "y" if orientation == "v" else "x"

    # Enable animation slider when granularity is None
    if granularity_param is None:
        args["animation_frame"] = "granularity"

    # Step 7: Determine the best granularity level to display initially
    best_granularity = _determine_best_granularity(
        combined_cumulative_df, granularity_param
    )

    # Step 8: Create the Plotly figure
    fig: go.Figure = make_figure(args, go.Scatter)

    # Step 9: Configure animation if granularity slider is enabled
    if granularity_param is None and hasattr(fig, "frames") and fig.frames:
        # Set initial frame to best granularity
        fig = go.Figure(
            data=fig.frames[best_granularity].data,
            frames=fig.frames,
            layout=fig.layout,
        )
        fig.layout["sliders"][0]["active"] = best_granularity

        # Configure fixed axis ranges to prevent jumping during animation
        _configure_animation_axes(
            fig, combined_cumulative_df, orientation or "v", best_granularity
        )

    return fig

"""Plotly histogram visualization with optimal binning.

This module provides a Plotly Express-compatible histogram function that uses
Khisto's optimal binning algorithm for automatic bin selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import pyarrow as pa
import narwhals as nw

from khisto.array import histogram_table
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
from plotly.basedatatypes import BasePlotlyType
import plotly.graph_objects as go

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoDataFrame, IntoSeries


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_histogram_for_groups(
    groups: dict,
    x_column_name: str,
    granularity: Optional[GranularityT],
    grouped_mappings: list,
) -> list[nw.DataFrame]:
    """Compute histogram data for each group in the data.

    Parameters
    ----------
    groups : dict
        Dictionary mapping group identifiers to DataFrames
    x_column_name : str
        Name of the column containing values to histogram
    granularity : int, 'best', or None
        Granularity level for histogram computation
    grouped_mappings : list
        List of grouping mappings from Plotly Express

    Returns
    -------
    list[nw.DataFrame]
        List of DataFrames containing histogram data for each group
    """
    histo_df_list: list[nw.DataFrame] = []

    for group_key, group_df in groups.items():
        # Determine the actual column name in this group's DataFrame
        value_column_name = x_column_name if x_column_name in group_df.columns else "x"

        # Compute histogram data for all granularities or specified level
        histo_df = histogram_table(group_df[value_column_name], granularity=granularity)

        # Add bin center and group identifier columns
        histo_df = (
            histo_df.with_columns(
                offsetgroup=nw.lit(", ".join(str(v) for v in group_key if v)),
            )
            .rename({"center": "x", "density": "y", "length": "width"})
            .with_columns(
                **{
                    mapping.grouper: nw.lit(value)
                    for mapping, value in zip(grouped_mappings, group_key)
                    if mapping is not None and mapping.grouper is not None
                }
            )
        )

        histo_df_list.append(histo_df)

    return histo_df_list


def _fill_missing_granularities(
    histo_df_list: list[nw.DataFrame],
) -> list[nw.DataFrame]:
    """Fill missing granularity levels across all groups.

    When different groups have different maximum granularities, this function
    duplicates the highest granularity level to fill in missing levels. This
    ensures all groups have the same granularity levels for synchronized
    animation.

    Parameters
    ----------
    histo_df_list : list[nw.DataFrame]
        List of histogram DataFrames, possibly with different max granularities

    Returns
    -------
    list[nw.DataFrame]
        List of histogram DataFrames with consistent granularity levels
    """
    if not histo_df_list:
        return histo_df_list

    # Find the maximum granularity across all groups
    max_granularity = max(df["granularity"].max() for df in histo_df_list)

    # Fill missing granularity levels for each group
    filled_list = []
    for histo_df in histo_df_list:
        actual_max_granularity = histo_df["granularity"].max()

        if actual_max_granularity < max_granularity:
            # Duplicate the highest granularity data for missing levels
            duplicates = [
                histo_df.filter(
                    nw.col("granularity") == actual_max_granularity
                ).with_columns(
                    granularity=nw.lit(g),
                    is_best=nw.lit(False),
                )
                for g in range(actual_max_granularity + 1, max_granularity + 1)
            ]
            filled_list.append(nw.concat([histo_df, *duplicates]))
        else:
            filled_list.append(histo_df)

    return filled_list


def _create_empty_histogram_dataframe() -> nw.DataFrame:
    """Create an empty histogram DataFrame with correct schema.

    Returns
    -------
    nw.DataFrame
        Empty DataFrame with histogram columns
    """
    return nw.from_dict(
        {
            "x": pa.array([]),
            "y": pa.array([]),
            "width": pa.array([]),
            "offsetgroup": pa.array([]),
            "granularity": pa.array([]),
        },
        backend="pyarrow",
    )


def _combine_histogram_dataframes(
    histo_df_list: list[nw.DataFrame],
) -> nw.DataFrame:
    """Combine histogram DataFrames from all groups into a single DataFrame.

    Parameters
    ----------
    histo_df_list : list[nw.DataFrame]
        List of histogram DataFrames for each group

    Returns
    -------
    nw.DataFrame
        Combined and sorted histogram DataFrame
    """
    if not histo_df_list:
        return _create_empty_histogram_dataframe()

    combined_df = nw.concat(histo_df_list)
    return combined_df.sort(["granularity", "offsetgroup", "x"])


def _determine_best_granularity(
    combined_histo_df: nw.DataFrame,
    granularity: Optional[GranularityT],
) -> int:
    """Determine the best granularity level to display initially.

    Parameters
    ----------
    combined_histo_df : nw.DataFrame
        Combined histogram DataFrame with all granularities
    granularity : int, 'best', or None
        User-specified granularity preference

    Returns
    -------
    int
        The granularity level to display initially
    """
    if isinstance(granularity, int):
        return granularity
    elif granularity == "best":
        return combined_histo_df["granularity"].max()
    else:  # granularity is None
        return combined_histo_df.filter(nw.col("is_best"))["granularity"].max()


def _configure_animation_axes(
    fig: go.Figure,
    combined_histo_df: nw.DataFrame,
    orientation: str,
    best_granularity: int,
) -> None:
    """Configure axis ranges for animated histograms.

    Sets fixed axis ranges based on the overall data range to prevent
    axes from jumping during animation.

    Parameters
    ----------
    fig : go.Figure
        The Plotly Figure to configure
    combined_histo_df : nw.DataFrame
        Combined histogram DataFrame with all granularities
    orientation : str
        'v' for vertical or 'h' for horizontal bars
    best_granularity : int
        The best granularity level for initial display
    """
    # Calculate data range with margin
    min_x = combined_histo_df["x"].min()
    max_x = combined_histo_df["x"].max()
    max_y = combined_histo_df["y"].max()
    axis_margin = 0.2 * (max_x - min_x)

    # Set appropriate axis ranges based on orientation
    if orientation == "v":
        fig.update_xaxes(
            range=(min_x - axis_margin, max_x + axis_margin),
            selector={"type": "bar"},
        )
        fig.update_yaxes(range=(0, max_y * 1.1), selector={"type": "bar"})
    else:
        fig.update_yaxes(
            range=(min_x - axis_margin, max_x + axis_margin),
            selector={"type": "bar"},
        )
        fig.update_xaxes(range=(0, max_y * 1.1), selector={"type": "bar"})


def _update_widths(
    df: nw.DataFrame,
    traces: list[BasePlotlyType],
    granularity: int,
    orientation: Literal["v", "h"],
    x_column_name: str,
    groups: dict,
) -> None:
    """Update bar widths for histogram traces based on bin sizes.

    This function sets the appropriate bar widths for histogram traces to match
    the variable-width bins computed by Khisto. It handles both bar traces
    (which get histogram bin widths) and other trace types like marginal plots
    (which get their original data positions).

    Parameters
    ----------
    df : nw.DataFrame
        Histogram DataFrame containing bin information for a specific granularity
        level. Must include columns: 'offsetgroup', 'granularity', 'width'
    traces : list[BasePlotlyType]
        List of Plotly trace objects to update
    granularity : int
        The granularity level being displayed
    orientation : {'v', 'h'}
        Histogram orientation - 'v' for vertical, 'h' for horizontal
    x_column_name : str
        Name of the column containing the original data values
    groups : dict
        Dictionary mapping group keys to their DataFrames, used for marginal plots

    Notes
    -----
    Bar traces get updated with variable widths from the histogram bins.
    Non-bar traces (e.g., rug or box plots) get updated with original data positions.
    """
    # Extract unique group identifiers in the correct order
    offsetgroups = df["offsetgroup"].unique(maintain_order=True).to_list()

    # Separate bar traces from marginal plot traces
    bar_traces = [trace for trace in traces if getattr(trace, "type", None) == "bar"]
    other_traces = [trace for trace in traces if getattr(trace, "type", None) != "bar"]

    # Update bar widths for histogram bars
    for trace, offsetgroup in zip(bar_traces, offsetgroups):
        # Filter to this specific group and granularity level
        group_df = df.filter(
            (nw.col("offsetgroup") == offsetgroup)
            & (nw.col("granularity") == granularity)
        )

        if len(group_df) > 0:
            widths = group_df["width"].to_list()
            trace.update(width=widths)

    # Update positions for marginal plots (rug, box, etc.)
    if other_traces:
        for trace, group_key in zip(other_traces, groups.keys()):
            # Determine the column name in this group's DataFrame
            value_column = (
                x_column_name if x_column_name in groups[group_key].columns else "x"
            )

            original_values = groups[group_key][value_column]

            # Set appropriate axis based on orientation
            if orientation == "v":
                trace.update(x=original_values)
            else:
                trace.update(y=original_values)


def histogram(
    data_frame: Optional[IntoDataFrame] = None,
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    color: Optional[str] = None,
    pattern_shape: Optional[str] = None,
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
    pattern_shape_sequence: Optional[list[str]] = None,
    pattern_shape_map: Optional[dict] = None,
    marginal: Optional[Literal["rug", "box"]] = None,
    opacity: Optional[float] = None,
    orientation: Optional[Literal["v", "h"]] = "v",
    barmode: Literal["relative", "overlay", "group"] = "relative",
    log_x: bool = False,
    log_y: bool = False,
    range_x: Optional[list] = None,
    range_y: Optional[list] = None,
    granularity: Optional[GranularityT] = "best",
    text_auto: Union[bool, str] = False,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    template: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> go.Figure:
    """Create a histogram using Khisto's optimal binning algorithm.

    This function provides a Plotly Express-compatible interface for creating
    histograms with automatic optimal bin selection using the Khisto algorithm.
    Unlike standard histograms that require manual bin specification, Khisto
    automatically determines the optimal number of bins and their boundaries
    to best represent the underlying data distribution.

    The function accepts data in multiple formats:

    - Direct arrays: NumPy arrays, lists, tuples
    - Pandas/Polars Series or DataFrames
    - Any array supporting the Python Array API standard

    When using DataFrames, additional Plotly Express features are available
    including color grouping, faceting, animations, and marginal plots.

    Parameters
    ----------
    data_frame : IntoDataFrame, optional
        A DataFrame-like object (pandas, polars, etc.) or a Narwhals DataFrame.
        Either `data_frame` or direct array input via `x` must be provided.
    x : str or ArrayT or IntoSeries, optional
        Either a column name in `data_frame`, or an array/Series for the values
        to bin. Supports NumPy arrays, pandas/polars Series, lists, tuples, and
        any array supporting the Python Array API standard.
    color : str, optional
        Column name in `data_frame` for color encoding. Creates separate histogram
        traces for each unique value in this column.
    pattern_shape : str, optional
        Column name in `data_frame` for pattern shape encoding. Adds different
        fill patterns to bars based on this column's values.
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
        animated histogram with a timeline slider.
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
        Example: {'x': 'Temperature (°C)', 'y': 'Density'}
    color_discrete_sequence : list of str, optional
        List of CSS color strings to cycle through for discrete color values.
        Example: ['#FF0000', '#00FF00', '#0000FF']
    color_discrete_map : dict, optional
        Dict mapping specific discrete color values to CSS colors. Provides
        complete control over which values get which colors.
        Example: {'A': 'red', 'B': 'blue', 'C': 'green'}
    pattern_shape_sequence : list of str, optional
        List of pattern shapes to cycle through. Available patterns include '',
        '/', '\\', 'x', '-', '|', '+', '.'.
    pattern_shape_map : dict, optional
        Dict mapping pattern shape values to specific patterns. Provides complete
        control over pattern assignment.
    marginal : {'rug', 'box'}, optional
        Type of marginal distribution plot to add along the axis.
        - 'rug': Shows individual data points as tick marks
        - 'box': Shows a box plot summarizing the distribution
    opacity : float, optional
        Opacity of histogram bars (0 to 1). Lower values create transparency,
        useful for overlapping histograms. Default is 1 (fully opaque).
    orientation : {'v', 'h'}, default 'v'
        Orientation of histogram.
        - 'v': Vertical bars (default) with x-axis showing values
        - 'h': Horizontal bars with y-axis showing values
    barmode : {'relative', 'overlay', 'group'}, default 'relative'
        How to display bars when using color encoding.
        - 'relative': Stack bars on top of each other (default)
        - 'overlay': Overlay bars (useful with transparency)
        - 'group': Group bars side by side
    log_x : bool, default False
        If True, use logarithmic scale for x-axis. Useful for data spanning
        multiple orders of magnitude.
    log_y : bool, default False
        If True, use logarithmic scale for y-axis (density/count axis).
    range_x : list, optional
        Range for x-axis as [min, max]. Restricts the visible range of the
        histogram. Example: [-3, 3]
    range_y : list, optional
        Range for y-axis as [min, max]. Restricts the visible range of the
        density axis. Example: [0, 0.5]
    granularity : int or 'best' or None, default 'best'
        Granularity level to use for histogram binning.
        - 'best': Uses the optimal granularity level (default)
        - int: Uses the specified granularity level
        - None: Creates an interactive slider to explore all granularity levels
        When None and multiple groups exist (via color, facets, etc.), the slider
        synchronizes across all groups to show the same granularity level.
    text_auto : bool or str, default False
        If True, automatically display density values on bars. If a format string
        (e.g., '.2f'), use that format for the text. Example: '.3f' for 3 decimals.
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
        A Plotly Figure object with the histogram visualization. The figure can
        be displayed using `fig.show()`, saved using `fig.write_html()` or
        `fig.write_image()`, or further customized using Plotly's API.

    Examples
    --------
    Create a simple histogram from a NumPy array:

    >>> import numpy as np
    >>> from khisto.plot.plotly import histogram
    >>> data = np.random.normal(0, 1, 1000)
    >>> fig = histogram(x=data)
    >>> fig.show()

    Create a histogram from a DataFrame with color grouping:

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'value': np.random.normal(0, 1, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000)
    ... })
    >>> fig = histogram(df, x='value', color='category')
    >>> fig.show()

    Create a horizontal histogram with custom styling:

    >>> fig = histogram(
    ...     x=data,
    ...     orientation='h',
    ...     title='Distribution of Values',
    ...     template='plotly_white',
    ...     opacity=0.7
    ... )
    >>> fig.show()

    Create a histogram with marginal rug plot:

    >>> fig = histogram(x=data, marginal='rug')
    >>> fig.show()

    Create overlaid histograms with transparency:

    >>> fig = histogram(
    ...     df,
    ...     x='value',
    ...     color='category',
    ...     barmode='overlay',
    ...     opacity=0.6
    ... )
    >>> fig.show()

    Create a histogram with custom axis ranges and labels:

    >>> fig = histogram(
    ...     x=data,
    ...     range_x=[-3, 3],
    ...     labels={'x': 'Standard Deviations', 'y': 'Probability Density'},
    ...     title='Normal Distribution'
    ... )
    >>> fig.show()

    See Also
    --------
    khisto.array.histogram : Compute histogram arrays (densities and bin edges)
    khisto.array.histogram_bin_edges : Compute only the bin edges
    khisto.array.histogram_series : Get full histogram information as DataFrame

    Notes
    -----
    This function uses the Khisto algorithm for automatic optimal bin selection.
    The Khisto algorithm analyzes the data distribution to determine the best
    binning strategy that maximizes information while minimizing complexity.

    Key features of Khisto's approach:

    - **Automatic bin selection**: No need to specify number of bins or bin width
    - **Density-based**: Y-axis represents probability density by default, making
      histograms comparable across different sample sizes
    - **Variable bin widths**: Bins can have different widths to better capture
      the data distribution (wider bins in sparse regions, narrower in dense regions)
    - **Multi-granularity**: Khisto can analyze data at multiple scales to find
      the optimal representation

    The histogram bars use Khisto's optimal bins with potentially variable widths.
    Each bar's height represents the probability density for that bin, computed as
    the proportion of data points divided by the bin width. This ensures the total
    area of all bars equals 1, making the histogram a proper probability density
    estimate.

    When using color grouping, faceting, or animations, Khisto computes optimal
    bins independently for each group, ensuring the best representation for each
    subset of data.
    """
    # Step 1: Prepare arguments and build dataframe structure
    args = locals().copy()
    granularity = args.pop("granularity")

    apply_default_cascade(args)
    args = build_dataframe(args, go.Bar)

    # Step 2: Infer plot configuration and group data
    trace_specs, grouped_mappings, sizeref, show_colorbar = infer_config(
        args.copy(), go.Bar, {}, {}
    )
    grouper = [x.grouper or one_group for x in grouped_mappings] or [one_group]
    groups, orders = get_groups_and_orders(args.copy(), grouper)

    # Step 3: Compute histogram data for each group
    x_column_name = "_value" if not isinstance(x, str) else args["x"]
    histo_df_list = _compute_histogram_for_groups(
        groups, x_column_name, granularity, grouped_mappings
    )

    # Step 4: Ensure consistent granularity levels across groups (for animation)
    if granularity is None:
        histo_df_list = _fill_missing_granularities(histo_df_list)

    # Step 5: Combine all histogram data into a single DataFrame
    combined_histo_df = _combine_histogram_dataframes(histo_df_list)

    # Step 6: Prepare data for Plotly figure creation
    args["data_frame"] = nw.to_native(combined_histo_df)
    args["x"] = "x" if orientation == "v" else "y"
    args["y"] = "y" if orientation == "v" else "x"

    # Enable animation slider when granularity is None
    if granularity is None:
        args["animation_frame"] = "granularity"

    # Step 7: Determine the best granularity level to display initially
    best_granularity = _determine_best_granularity(combined_histo_df, granularity)

    # Step 8: Create the Plotly figure
    fig: go.Figure = make_figure(args, go.Bar)

    # Step 9: Configure animation if granularity slider is enabled
    if granularity is None:
        # Set initial frame to best granularity
        fig = go.Figure(
            data=fig.frames[best_granularity].data,
            frames=fig.frames,
            layout=fig.layout,
        )
        fig.layout["sliders"][0]["active"] = best_granularity

        # Configure fixed axis ranges to prevent jumping during animation
        _configure_animation_axes(
            fig, combined_histo_df, orientation or "v", best_granularity
        )

    # Step 10: Update bar widths for the initial display
    _update_widths(
        combined_histo_df.filter(nw.col("granularity") == best_granularity),
        fig.data,
        granularity=best_granularity,
        orientation=orientation if orientation else "v",
        x_column_name=x_column_name,
        groups=groups,
    )

    # Step 11: Update bar widths for all animation frames (if applicable)
    if hasattr(fig, "frames") and fig.frames:
        for granularity_level, frame in enumerate(fig.frames):
            _update_widths(
                combined_histo_df.filter(nw.col("granularity") == granularity_level),
                frame.data,
                granularity=granularity_level,
                orientation=orientation if orientation else "v",
                x_column_name=x_column_name,
                groups=groups,
            )

    return fig

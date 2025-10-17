from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union


from khisto.array import histogram_series
import narwhals as nw
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
    from khisto.typing import ArrayT
    from narwhals.typing import IntoDataFrame, IntoSeries


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
    cumulative: bool = False,
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
    cumulative : bool, default False
        If True, create cumulative histogram where each bar shows the cumulative
        sum/density up to that point.
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

    Create a cumulative distribution plot:

    >>> fig = histogram(x=data, cumulative=True)
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
    args = locals().copy()

    apply_default_cascade(args)
    args = build_dataframe(args, go.Bar)
    trace_specs, grouped_mappings, sizeref, show_colorbar = infer_config(
        args.copy(), go.Bar, {}, {}
    )
    grouper = [x.grouper or one_group for x in grouped_mappings] or [one_group]
    groups, orders = get_groups_and_orders(args.copy(), grouper)

    histo_df_list = []
    x_column_name = "_value" if not isinstance(x, str) else args["x"]
    for g, g_df in groups.items():
        value_column_name = x_column_name
        if value_column_name not in g_df.columns:
            value_column_name = "x"
        histo_df = histogram_series(g_df[value_column_name], only_best=True)
        histo_df = (
            histo_df.with_columns(
                center=((histo_df["lower_bound"] + histo_df["upper_bound"]) / 2),
                offsetgroup=nw.lit(", ".join(str(v) for v in g if v)),
            )
            .rename(
                {
                    "center": "x",
                    "density": "y",
                    "length": "width",
                }
            )
            .select(["x", "y", "width", "offsetgroup"])
            .with_columns(
                **{
                    k.grouper: nw.lit(v)
                    for k, v in zip(grouped_mappings, g)
                    if k is not None and k.grouper is not None
                }
            )
        )

        histo_df_list.append(histo_df)

    if not histo_df_list:
        combined_histo_df: nw.DataFrame = nw.from_dict(
            {"x": [], "y": [], "width": [], "offsetgroup": []}, backend="pyarrow"
        )
    else:
        combined_histo_df: nw.DataFrame = nw.concat(histo_df_list)

    args["data_frame"] = nw.to_native(combined_histo_df)
    args["x"] = "x" if orientation == "v" else "y"
    args["y"] = "y" if orientation == "v" else "x"

    offsetgroups = combined_histo_df["offsetgroup"].unique().to_list()

    # Use make_figure with our Histogram constructor
    fig = make_figure(args, go.Bar)

    bar_traces = [trace for trace in fig.data if trace.type == "bar"]
    other_traces = [trace for trace in fig.data if trace.type != "bar"]

    # Update bar widths for histogram bars
    for trace, offsetgroup in zip(bar_traces, offsetgroups):
        mask = combined_histo_df["offsetgroup"] == offsetgroup
        widths = combined_histo_df.filter(mask)["width"].to_list()
        trace.update(width=widths)

    # Update marginal traces with original data
    for trace, offsetgroup in zip(other_traces, groups.keys()):
        value_column_name = x_column_name
        if value_column_name not in groups[offsetgroup].columns:
            value_column_name = "x"
        original_x = groups[offsetgroup][value_column_name]
        trace.update(x=original_x if orientation == "v" else None)
        trace.update(y=original_x if orientation == "h" else None)

    return fig

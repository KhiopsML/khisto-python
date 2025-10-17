from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union


from khisto.array import histogram_series
import narwhals as nw

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
    import plotly.graph_objects as go


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
    When `nbins` is not specified, Khisto automatically determines the optimal
    number of bins and their boundaries.

    Parameters
    ----------
    data_frame : IntoDataFrame, optional
        A DataFrame-like object (pandas, polars, etc.) or a Narwhals DataFrame.
        Either `data_frame` or direct array input via `x` must be provided.
    x : str or ArrayT or IntoSeries, optional
        Either a column name in `data_frame`, or an array/Series for the values to bin.
    color : str, optional
        Column name in `data_frame` for color encoding.
    pattern_shape : str, optional
        Column name in `data_frame` for pattern shape encoding.
    facet_row : str, optional
        Column name in `data_frame` for faceting into subplot rows.
    facet_col : str, optional
        Column name in `data_frame` for faceting into subplot columns.
    facet_col_wrap : int, optional
        Maximum number of facet columns. Wraps to new rows if exceeded.
    facet_row_spacing : float, optional
        Spacing between facet rows (0 to 1).
    facet_col_spacing : float, optional
        Spacing between facet columns (0 to 1).
    hover_name : str, optional
        Column name in `data_frame` for bold hover tooltip text.
    hover_data : list of str or dict, optional
        Columns to include in hover tooltips. Can be a list of column names
        or a dict mapping column names to formatting strings.
    animation_frame : str, optional
        Column name in `data_frame` for animation frame grouping.
    animation_group : str, optional
        Column name in `data_frame` for matching objects across frames.
    category_orders : dict, optional
        Dict mapping categorical column names to ordered lists of values.
    labels : dict, optional
        Dict mapping column names to custom axis/legend labels.
    color_discrete_sequence : list of str, optional
        List of CSS color strings to cycle through for discrete color values.
    color_discrete_map : dict, optional
        Dict mapping discrete color values to specific CSS colors.
    pattern_shape_sequence : list of str, optional
        List of pattern shapes to cycle through.
    pattern_shape_map : dict, optional
        Dict mapping pattern shape values to specific patterns.
    marginal : {'rug', 'box', 'violin', 'histogram'}, optional
        Type of marginal distribution plot to add.
    opacity : float, optional
        Opacity of bars (0 to 1).
    orientation : {'v', 'h'}, optional
        Orientation of histogram. 'v' for vertical (default), 'h' for horizontal.
    barmode : {'relative', 'overlay', 'group'}, default 'relative'
        How to display bars when using color encoding.
        - 'relative': Stack bars
        - 'overlay': Overlay bars
        - 'group': Group bars side by side
    log_x : bool, default False
        Use log scale for x-axis.
    log_y : bool, default False
        Use log scale for y-axis.
    range_x : list, optional
        Range for x-axis as [min, max].
    range_y : list, optional
        Range for y-axis as [min, max].
    cumulative : bool, default False
        If True, create cumulative histogram.
    text_auto : bool or str, default False
        If True or a format string, display values on bars.
    title : str, optional
        Figure title.
    template : str, optional
        Plotly template name (e.g., 'plotly', 'plotly_white', 'ggplot2').
    width : int, optional
        Figure width in pixels.
    height : int, optional
        Figure height in pixels.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the histogram visualization.

    Examples
    --------
    Create a simple histogram from an array:

    >>> import numpy as np
    >>> from khisto.plot.plotly import histogram
    >>> data = np.random.normal(0, 1, 1000)
    >>> fig = histogram(x=data)
    >>> fig.show()

    Create a histogram from a DataFrame with color grouping:

    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'value': np.random.normal(0, 1, 1000),
    ...     'category': np.random.choice(['A', 'B', 'C'], 1000)
    ... })
    >>> fig = histogram(df, x='value', color='category')
    >>> fig.show()

    Create a horizontal histogram:

    >>> fig = histogram(x=data, orientation='h')
    >>> fig.show()

    Notes
    -----
    This function uses the Khisto algorithm for automatic optimal bin selection.
    The Khisto algorithm analyzes the data distribution to determine the best
    binning strategy that maximizes information while minimizing complexity.
    The histogram uses Khisto's density values by default.
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

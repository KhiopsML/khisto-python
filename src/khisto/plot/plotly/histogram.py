from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union

import narwhals as nw

from khisto.array import histogram_series
from khisto.utils import parse_narwhals_series

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
    marginal: Optional[Literal["rug", "box", "violin", "histogram"]] = None,
    opacity: Optional[float] = None,
    orientation: Optional[Literal["v", "h"]] = None,
    barmode: Literal["relative", "overlay", "group"] = "relative",
    log_x: bool = False,
    log_y: bool = False,
    range_x: Optional[list] = None,
    range_y: Optional[list] = None,
    cumulative: bool = False,
    text_auto: Union[bool, str] = False,
    title: Optional[str] = None,
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
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as e:
        raise ImportError(
            "plotly is required for this function. Install it with: pip install plotly"
        ) from e

    # Default to vertical orientation
    if orientation is None:
        orientation = "v"

    # Handle input data
    if data_frame is not None:
        # Convert to Narwhals for consistency
        df_nw = nw.from_native(data_frame)

        if x is None:
            raise ValueError("Column name 'x' must be provided when using data_frame")

        if not isinstance(x, str):
            raise ValueError(
                "When using data_frame, 'x' must be a column name (string)"
            )

        return _create_histogram_from_dataframe(
            df_nw=df_nw,
            x=x,
            color=color,
            pattern_shape=pattern_shape,
            facet_row=facet_row,
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=facet_row_spacing,
            facet_col_spacing=facet_col_spacing,
            hover_name=hover_name,
            hover_data=hover_data,
            animation_frame=animation_frame,
            animation_group=animation_group,
            category_orders=category_orders,
            labels=labels,
            color_discrete_sequence=color_discrete_sequence,
            color_discrete_map=color_discrete_map,
            pattern_shape_sequence=pattern_shape_sequence,
            pattern_shape_map=pattern_shape_map,
            marginal=marginal,
            opacity=opacity,
            orientation=orientation,
            barmode=barmode,
            log_x=log_x,
            log_y=log_y,
            range_x=range_x,
            range_y=range_y,
            cumulative=cumulative,
            text_auto=text_auto,
            title=title,
            template=template,
            width=width,
            height=height,
            go=go,
            make_subplots=make_subplots,
        )

    else:
        # Direct array input - use Khisto for optimal binning
        if x is None:
            raise ValueError("Either data_frame or x must be provided")

        # Parse narwhals series if applicable
        series = parse_narwhals_series(x)
        if series is not None:
            data = series
        else:
            data = x

        return _create_histogram_from_array(
            data=data,
            labels=labels,
            opacity=opacity,
            orientation=orientation,
            log_x=log_x,
            log_y=log_y,
            range_x=range_x,
            range_y=range_y,
            cumulative=cumulative,
            text_auto=text_auto,
            title=title,
            template=template,
            width=width,
            height=height,
            go=go,
        )


def _create_histogram_from_array(
    data,
    labels,
    opacity,
    orientation,
    log_x,
    log_y,
    range_x,
    range_y,
    cumulative,
    text_auto,
    title,
    template,
    width,
    height,
    go,
):
    """Create histogram from array data using Khisto bins."""
    # Use Khisto to compute histogram
    hist_df = histogram_series(data, only_best=True)

    # Extract histogram data using Khisto's computed values
    lower_bounds = hist_df["lower_bound"].to_list()
    upper_bounds = hist_df["upper_bound"].to_list()
    densities = hist_df["density"].to_list()
    bin_widths = hist_df["length"].to_list()

    # Compute bin centers from lower and upper bounds
    bin_centers = [
        (lower_bounds[i] + upper_bounds[i]) / 2 for i in range(len(lower_bounds))
    ]

    # Use density values from Khisto
    values = densities
    value_label = "Density"

    # Create cumulative if requested
    if cumulative:
        cumsum = 0
        cumulative_values = []
        for val in values:
            cumsum += val
            cumulative_values.append(cumsum)
        values = cumulative_values

    # Create the figure
    fig = go.Figure()

    if orientation == "v":
        # Vertical bars
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=values,
                width=bin_widths,
                opacity=opacity,
                text=[round(v, 3) for v in values] if text_auto else None,
                texttemplate=text_auto if isinstance(text_auto, str) else None,
                textposition="auto" if text_auto else None,
                hovertemplate=(
                    "Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>"
                    + f"{value_label}: %{{y}}<br>"
                    + "<extra></extra>"
                ),
                customdata=[
                    [lower_bounds[i], upper_bounds[i]] for i in range(len(lower_bounds))
                ],
            )
        )

        # Update layout
        fig.update_layout(
            xaxis_title=labels.get("x") if labels else None,
            yaxis_title=value_label if not labels else labels.get("y", value_label),
            xaxis_type="log" if log_x else None,
            yaxis_type="log" if log_y else None,
            xaxis_range=range_x,
            yaxis_range=range_y,
            title=title,
            template=template,
            width=width,
            height=height,
            bargap=0,
            bargroupgap=0,
        )
    else:
        # Horizontal bars
        fig.add_trace(
            go.Bar(
                y=bin_centers,
                x=values,
                width=bin_widths,
                orientation="h",
                opacity=opacity,
                text=[round(v, 3) for v in values] if text_auto else None,
                texttemplate=text_auto if isinstance(text_auto, str) else None,
                textposition="auto" if text_auto else None,
                hovertemplate=(
                    "Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>"
                    + f"{value_label}: %{{x}}<br>"
                    + "<extra></extra>"
                ),
                customdata=[
                    [lower_bounds[i], upper_bounds[i]] for i in range(len(lower_bounds))
                ],
            )
        )

        # Update layout
        fig.update_layout(
            yaxis_title=labels.get("x") if labels else None,
            xaxis_title=value_label if not labels else labels.get("y", value_label),
            yaxis_type="log" if log_y else None,
            xaxis_type="log" if log_x else None,
            yaxis_range=range_x,
            xaxis_range=range_y,
            title=title,
            template=template,
            width=width,
            height=height,
            bargap=0,
            bargroupgap=0,
        )

    return fig


def _create_histogram_from_dataframe(
    df_nw,
    x,
    color,
    pattern_shape,
    facet_row,
    facet_col,
    facet_col_wrap,
    facet_row_spacing,
    facet_col_spacing,
    hover_name,
    hover_data,
    animation_frame,
    animation_group,
    category_orders,
    labels,
    color_discrete_sequence,
    color_discrete_map,
    pattern_shape_sequence,
    pattern_shape_map,
    marginal,
    opacity,
    orientation,
    barmode,
    log_x,
    log_y,
    range_x,
    range_y,
    cumulative,
    text_auto,
    title,
    template,
    width,
    height,
    go,
    make_subplots,
):
    """Create histogram from DataFrame using Khisto bins with support for grouping and faceting."""
    # Get the column data
    col_data = df_nw[x]

    # Handle color grouping
    if color is not None:
        return _create_grouped_histogram(
            df_nw=df_nw,
            x=x,
            color=color,
            hover_name=hover_name,
            hover_data=hover_data,
            category_orders=category_orders,
            labels=labels,
            color_discrete_sequence=color_discrete_sequence,
            color_discrete_map=color_discrete_map,
            opacity=opacity,
            orientation=orientation,
            barmode=barmode,
            log_x=log_x,
            log_y=log_y,
            range_x=range_x,
            range_y=range_y,
            cumulative=cumulative,
            text_auto=text_auto,
            title=title,
            template=template,
            width=width,
            height=height,
            go=go,
        )

    # Handle faceting
    if facet_row is not None or facet_col is not None:
        return _create_faceted_histogram(
            df_nw=df_nw,
            x=x,
            facet_row=facet_row,
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
            facet_row_spacing=facet_row_spacing,
            facet_col_spacing=facet_col_spacing,
            hover_name=hover_name,
            hover_data=hover_data,
            category_orders=category_orders,
            labels=labels,
            opacity=opacity,
            orientation=orientation,
            log_x=log_x,
            log_y=log_y,
            range_x=range_x,
            range_y=range_y,
            cumulative=cumulative,
            text_auto=text_auto,
            title=title,
            template=template,
            width=width,
            height=height,
            go=go,
            make_subplots=make_subplots,
        )

    # Simple histogram without grouping or faceting
    hist_df = histogram_series(col_data, only_best=True)

    # Extract histogram data
    lower_bounds = hist_df["lower_bound"].to_list()
    upper_bounds = hist_df["upper_bound"].to_list()
    densities = hist_df["density"].to_list()
    bin_widths = hist_df["length"].to_list()

    # Compute bin centers
    bin_centers = [
        (lower_bounds[i] + upper_bounds[i]) / 2 for i in range(len(lower_bounds))
    ]

    values = densities
    value_label = "Density"

    # Create cumulative if requested
    if cumulative:
        cumsum = 0
        cumulative_values = []
        for val in values:
            cumsum += val
            cumulative_values.append(cumsum)
        values = cumulative_values

    # Build hover template
    hover_template_parts = []
    if hover_name and hover_name in df_nw.columns:
        hover_template_parts.append(f"<b>{hover_name}: %{{customdata[2]}}</b><br>")

    hover_template_parts.append("Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>")

    if orientation == "v":
        hover_template_parts.append(f"{value_label}: %{{y}}<br>")
    else:
        hover_template_parts.append(f"{value_label}: %{{x}}<br>")

    if hover_data:
        hover_cols = (
            hover_data if isinstance(hover_data, list) else list(hover_data.keys())
        )
        for idx, col in enumerate(hover_cols):
            if col in df_nw.columns:
                hover_template_parts.append(f"{col}: %{{customdata[{idx + 3}]}}<br>")

    hover_template_parts.append("<extra></extra>")

    # Prepare custom data
    customdata = [[lower_bounds[i], upper_bounds[i]] for i in range(len(lower_bounds))]

    # Create the figure
    fig = go.Figure()

    if orientation == "v":
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=values,
                width=bin_widths,
                opacity=opacity,
                text=[round(v, 3) for v in values] if text_auto else None,
                texttemplate=text_auto if isinstance(text_auto, str) else None,
                textposition="auto" if text_auto else None,
                hovertemplate="".join(hover_template_parts),
                customdata=customdata,
            )
        )

        fig.update_layout(
            xaxis_title=labels.get("x", x) if labels else x,
            yaxis_title=value_label if not labels else labels.get("y", value_label),
            xaxis_type="log" if log_x else None,
            yaxis_type="log" if log_y else None,
            xaxis_range=range_x,
            yaxis_range=range_y,
            title=title,
            template=template,
            width=width,
            height=height,
            bargap=0,
            bargroupgap=0,
        )
    else:
        fig.add_trace(
            go.Bar(
                y=bin_centers,
                x=values,
                width=bin_widths,
                orientation="h",
                opacity=opacity,
                text=[round(v, 3) for v in values] if text_auto else None,
                texttemplate=text_auto if isinstance(text_auto, str) else None,
                textposition="auto" if text_auto else None,
                hovertemplate="".join(hover_template_parts),
                customdata=customdata,
            )
        )

        fig.update_layout(
            yaxis_title=labels.get("x", x) if labels else x,
            xaxis_title=value_label if not labels else labels.get("y", value_label),
            yaxis_type="log" if log_y else None,
            xaxis_type="log" if log_x else None,
            yaxis_range=range_x,
            xaxis_range=range_y,
            title=title,
            template=template,
            width=width,
            height=height,
            bargap=0,
            bargroupgap=0,
        )

    return fig


def _create_grouped_histogram(
    df_nw,
    x,
    color,
    hover_name,
    hover_data,
    category_orders,
    labels,
    color_discrete_sequence,
    color_discrete_map,
    opacity,
    orientation,
    barmode,
    log_x,
    log_y,
    range_x,
    range_y,
    cumulative,
    text_auto,
    title,
    template,
    width,
    height,
    go,
):
    """Create histogram with color grouping using Khisto bins."""
    # Get unique groups
    groups = df_nw[color].unique().to_list()

    # Apply category orders if specified
    if category_orders and color in category_orders:
        groups = [g for g in category_orders[color] if g in groups]

    # Setup colors
    if color_discrete_map:
        colors = [color_discrete_map.get(g) for g in groups]
    elif color_discrete_sequence:
        colors = [
            color_discrete_sequence[i % len(color_discrete_sequence)]
            for i in range(len(groups))
        ]
    else:
        # Default plotly colors
        default_colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        colors = [default_colors[i % len(default_colors)] for i in range(len(groups))]

    fig = go.Figure()
    value_label = "Density"

    for idx, group in enumerate(groups):
        # Filter data for this group
        group_data = df_nw.filter(nw.col(color) == group)[x]

        # Compute histogram for this group
        hist_df = histogram_series(group_data, only_best=True)

        lower_bounds = hist_df["lower_bound"].to_list()
        upper_bounds = hist_df["upper_bound"].to_list()
        densities = hist_df["density"].to_list()
        bin_widths = hist_df["length"].to_list()

        bin_centers = [
            (lower_bounds[i] + upper_bounds[i]) / 2 for i in range(len(lower_bounds))
        ]

        values = densities

        if cumulative:
            cumsum = 0
            cumulative_values = []
            for val in values:
                cumsum += val
                cumulative_values.append(cumsum)
            values = cumulative_values

        # Create trace
        if orientation == "v":
            fig.add_trace(
                go.Bar(
                    x=bin_centers,
                    y=values,
                    width=bin_widths,
                    name=str(group),
                    marker_color=colors[idx] if colors[idx] else None,
                    opacity=opacity,
                    text=[round(v, 3) for v in values] if text_auto else None,
                    texttemplate=text_auto if isinstance(text_auto, str) else None,
                    textposition="auto" if text_auto else None,
                    hovertemplate=(
                        f"<b>{color}: {group}</b><br>"
                        + "Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>"
                        + f"{value_label}: %{{y}}<br>"
                        + "<extra></extra>"
                    ),
                    customdata=[
                        [lower_bounds[i], upper_bounds[i]]
                        for i in range(len(lower_bounds))
                    ],
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    y=bin_centers,
                    x=values,
                    width=bin_widths,
                    orientation="h",
                    name=str(group),
                    marker_color=colors[idx] if colors[idx] else None,
                    opacity=opacity,
                    text=[round(v, 3) for v in values] if text_auto else None,
                    texttemplate=text_auto if isinstance(text_auto, str) else None,
                    textposition="auto" if text_auto else None,
                    hovertemplate=(
                        f"<b>{color}: {group}</b><br>"
                        + "Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>"
                        + f"{value_label}: %{{x}}<br>"
                        + "<extra></extra>"
                    ),
                    customdata=[
                        [lower_bounds[i], upper_bounds[i]]
                        for i in range(len(lower_bounds))
                    ],
                )
            )

    # Update layout
    if orientation == "v":
        fig.update_layout(
            xaxis_title=labels.get("x", x) if labels else x,
            yaxis_title=value_label if not labels else labels.get("y", value_label),
            xaxis_type="log" if log_x else None,
            yaxis_type="log" if log_y else None,
            xaxis_range=range_x,
            yaxis_range=range_y,
            title=title,
            template=template,
            width=width,
            height=height,
            barmode=barmode,
            bargap=0,
            bargroupgap=0,
        )
    else:
        fig.update_layout(
            yaxis_title=labels.get("x", x) if labels else x,
            xaxis_title=value_label if not labels else labels.get("y", value_label),
            yaxis_type="log" if log_y else None,
            xaxis_type="log" if log_x else None,
            yaxis_range=range_x,
            xaxis_range=range_y,
            title=title,
            template=template,
            width=width,
            height=height,
            barmode=barmode,
            bargap=0,
            bargroupgap=0,
        )

    return fig


def _create_faceted_histogram(
    df_nw,
    x,
    facet_row,
    facet_col,
    facet_col_wrap,
    facet_row_spacing,
    facet_col_spacing,
    hover_name,
    hover_data,
    category_orders,
    labels,
    opacity,
    orientation,
    log_x,
    log_y,
    range_x,
    range_y,
    cumulative,
    text_auto,
    title,
    template,
    width,
    height,
    go,
    make_subplots,
):
    """Create faceted histogram using Khisto bins."""
    # Get unique facet values
    row_facets = [None]
    col_facets = [None]

    if facet_row:
        row_facets = df_nw[facet_row].unique().to_list()
        if category_orders and facet_row in category_orders:
            row_facets = [f for f in category_orders[facet_row] if f in row_facets]

    if facet_col:
        col_facets = df_nw[facet_col].unique().to_list()
        if category_orders and facet_col in category_orders:
            col_facets = [f for f in category_orders[facet_col] if f in col_facets]

    # Handle col_wrap
    n_rows = len(row_facets)
    n_cols = len(col_facets)

    if facet_col_wrap and facet_col and not facet_row:
        n_cols = min(facet_col_wrap, len(col_facets))
        n_rows = (len(col_facets) + n_cols - 1) // n_cols

    # Create subplots
    subplot_titles = []
    for row_val in row_facets:
        for col_val in col_facets:
            parts = []
            if facet_row and row_val is not None:
                parts.append(f"{facet_row}={row_val}")
            if facet_col and col_val is not None:
                parts.append(f"{facet_col}={col_val}")
            subplot_titles.append(", ".join(parts) if parts else "")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles if subplot_titles[0] else None,
        vertical_spacing=facet_row_spacing if facet_row_spacing else 0.03,
        horizontal_spacing=facet_col_spacing if facet_col_spacing else 0.03,
    )

    value_label = "Density"

    for row_idx, row_val in enumerate(row_facets, 1):
        for col_idx, col_val in enumerate(col_facets, 1):
            # Filter data
            mask = nw.lit(True)
            if facet_row and row_val is not None:
                mask = mask & (nw.col(facet_row) == row_val)
            if facet_col and col_val is not None:
                mask = mask & (nw.col(facet_col) == col_val)

            facet_data = df_nw.filter(mask)[x]

            if len(facet_data) == 0:
                continue

            # Compute histogram
            hist_df = histogram_series(facet_data, only_best=True)

            lower_bounds = hist_df["lower_bound"].to_list()
            upper_bounds = hist_df["upper_bound"].to_list()
            densities = hist_df["density"].to_list()
            bin_widths = hist_df["length"].to_list()

            bin_centers = [
                (lower_bounds[i] + upper_bounds[i]) / 2
                for i in range(len(lower_bounds))
            ]

            values = densities

            if cumulative:
                cumsum = 0
                cumulative_values = []
                for val in values:
                    cumsum += val
                    cumulative_values.append(cumsum)
                values = cumulative_values

            # Add trace
            if orientation == "v":
                fig.add_trace(
                    go.Bar(
                        x=bin_centers,
                        y=values,
                        width=bin_widths,
                        opacity=opacity,
                        showlegend=False,
                        hovertemplate=(
                            "Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>"
                            + f"{value_label}: %{{y}}<br>"
                            + "<extra></extra>"
                        ),
                        customdata=[
                            [lower_bounds[i], upper_bounds[i]]
                            for i in range(len(lower_bounds))
                        ],
                    ),
                    row=row_idx,
                    col=col_idx,
                )
            else:
                fig.add_trace(
                    go.Bar(
                        y=bin_centers,
                        x=values,
                        width=bin_widths,
                        orientation="h",
                        opacity=opacity,
                        showlegend=False,
                        hovertemplate=(
                            "Bin: [%{customdata[0]:.4g}, %{customdata[1]:.4g})<br>"
                            + f"{value_label}: %{{x}}<br>"
                            + "<extra></extra>"
                        ),
                        customdata=[
                            [lower_bounds[i], upper_bounds[i]]
                            for i in range(len(lower_bounds))
                        ],
                    ),
                    row=row_idx,
                    col=col_idx,
                )

    # Update layout
    fig.update_layout(
        title=title,
        template=template,
        width=width,
        height=height,
        bargap=0,
        bargroupgap=0,
    )

    # Update axes
    if orientation == "v":
        fig.update_xaxes(
            title_text=labels.get("x", x) if labels else x,
            type="log" if log_x else None,
            range=range_x,
        )
        fig.update_yaxes(
            title_text=value_label if not labels else labels.get("y", value_label),
            type="log" if log_y else None,
            range=range_y,
        )
    else:
        fig.update_yaxes(
            title_text=labels.get("x", x) if labels else x,
            type="log" if log_y else None,
            range=range_x,
        )
        fig.update_xaxes(
            title_text=value_label if not labels else labels.get("y", value_label),
            type="log" if log_x else None,
            range=range_y,
        )

    return fig

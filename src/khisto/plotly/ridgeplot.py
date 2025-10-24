"""Plotly ridge plot visualization with optimal binning.

This module provides a ridge plot (joy plot) function that uses Khisto's
optimal binning algorithm for automatic bin selection. Ridge plots display
multiple distributions stacked vertically with partial overlap, making them
ideal for comparing distributions across categories or time periods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

import pyarrow as pa
import narwhals as nw

from khisto.array import histogram_series
from khisto.utils._compat._optional import import_optional_dependency, Extras

import_optional_dependency("plotly", extra=Extras.PLOTLY, errors="raise")
import plotly.graph_objects as go
from plotly.express._core import (
    build_dataframe,
    apply_default_cascade,
    infer_config,
    get_groups_and_orders,
    one_group,
)

if TYPE_CHECKING:
    from khisto.typing import GranularityT


# ============================================================================
# Helper Functions
# ============================================================================


def _compute_ridge_histogram_for_groups(
    groups: dict,
    x_column_name: str,
    granularity: Optional[GranularityT],
    grouped_mappings: list,
) -> list[nw.DataFrame]:
    """Compute histogram data for each group in ridge plot.

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
        List of DataFrames containing ridge histogram data for each group
    """
    histo_df_list: list[nw.DataFrame] = []

    for group_key, group_df in groups.items():
        # Determine the actual column name in this group's DataFrame
        value_column_name = x_column_name if x_column_name in group_df.columns else "x"

        if len(group_df) == 0:
            continue

        # Compute histogram for this category
        histo_df = histogram_series(
            group_df[value_column_name], granularity=granularity
        )

        # Convert group_key to string (it's usually a tuple from plotly grouping)
        group_key_str = (
            str(group_key[0])
            if isinstance(group_key, tuple) and len(group_key) == 1
            else str(group_key)
        )

        # Create ridge plot DataFrame with offsetgroup for this category
        ridge_df = histo_df.with_columns(
            [
                nw.col("lower_bound").alias("lower_bound"),
                nw.col("upper_bound").alias("upper_bound"),
                ((nw.col("lower_bound") + nw.col("upper_bound")) / 2).alias("x"),
                nw.col("density").alias("y"),
                nw.lit(group_key_str).alias("offsetgroup"),
            ]
        )

        histo_df_list.append(ridge_df)

    return histo_df_list


def _create_empty_ridge_dataframe() -> nw.DataFrame:
    """Create an empty ridge plot DataFrame with correct schema.

    Returns
    -------
    nw.DataFrame
        Empty DataFrame with required columns
    """
    return nw.from_dict(
        {
            "x": pa.array([]),
            "y": pa.array([]),
            "lower_bound": pa.array([]),
            "upper_bound": pa.array([]),
            "offsetgroup": pa.array([]),
            "granularity": pa.array([]),
        },
        backend="pyarrow",
    )


def _fill_missing_granularities_ridge(
    histo_df_list: list[nw.DataFrame],
) -> list[nw.DataFrame]:
    """Fill missing granularity levels across all categories.

    When different categories have different maximum granularities, this function
    duplicates the highest granularity level to fill in missing levels.

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


def _combine_ridge_dataframes(
    histo_df_list: list[nw.DataFrame],
) -> nw.DataFrame:
    """Combine ridge DataFrames from all groups into a single DataFrame.

    Parameters
    ----------
    histo_df_list : list[nw.DataFrame]
        List of ridge DataFrames for each group

    Returns
    -------
    nw.DataFrame
        Combined and sorted ridge DataFrame
    """
    if not histo_df_list:
        return _create_empty_ridge_dataframe()

    combined_df = nw.concat(histo_df_list)
    return combined_df.sort(["granularity", "offsetgroup", "x"])


def _determine_best_granularity_ridge(
    histo_df: nw.DataFrame,
    granularity: Optional[GranularityT],
) -> int:
    """Determine the best granularity level to display initially.

    Parameters
    ----------
    histo_df : nw.DataFrame
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
        return histo_df["granularity"].max()
    else:  # granularity is None
        return histo_df.filter(nw.col("is_best"))["granularity"].max()


def _create_ridge_traces(
    histo_df: nw.DataFrame,
    granularity_level: int,
    category_order: list,
    overlap: float,
    colors: Optional[list[str]],
    fill_colors: Optional[list[str]],
    line_width: float,
    line_color: str,
    show_legend: bool,
) -> list[go.Scatter]:
    """Create Plotly Scatter traces for the ridge plot.

    Parameters
    ----------
    histo_df : nw.DataFrame
        Histogram data for all categories (combined DataFrame)
    granularity_level : int
        Granularity level to visualize
    category_order : list
        Ordered list of categories (offsetgroups)
    overlap : float
        Vertical spacing overlap factor
    colors : list of str, optional
        Line colors for each category (not used when line_color is provided)
    fill_colors : list of str, optional
        Fill colors for each category
    line_width : float
        Width of the distribution lines
    line_color : str
        Color of the distribution outline
    show_legend : bool
        Whether to show legend entries

    Returns
    -------
    list[go.Scatter]
        List of Scatter traces for the ridge plot
    """
    traces = []

    # Filter to the specified granularity level
    # If a category doesn't have data at this level, use its maximum available granularity
    plot_df_dict = {}
    for offsetgroup in category_order:
        # Try to get data at the requested granularity
        category_at_level = histo_df.filter(
            (nw.col("offsetgroup") == offsetgroup)
            & (nw.col("granularity") == granularity_level)
        )

        if len(category_at_level) == 0:
            # This category doesn't have data at this granularity level
            # Use the maximum available granularity for this category
            category_data = histo_df.filter(nw.col("offsetgroup") == offsetgroup)
            if len(category_data) > 0:
                max_available_granularity = category_data["granularity"].max()
                category_at_level = histo_df.filter(
                    (nw.col("offsetgroup") == offsetgroup)
                    & (nw.col("granularity") == max_available_granularity)
                )

        if len(category_at_level) > 0:
            plot_df_dict[offsetgroup] = category_at_level

    # Combine all category data
    if not plot_df_dict:
        return traces

    plot_df = nw.concat(list(plot_df_dict.values()))

    # Calculate the maximum density across all categories to scale the vertical offset
    max_density = plot_df["y"].max()

    # Vertical spacing: each ridge gets offset by max_density * (1 - overlap)
    # This ensures ridges are separated while allowing controlled overlap
    vertical_spacing = max_density * (1.0 - overlap)

    # Iterate in reverse order so that the first category is drawn last (on top)
    # This creates the proper ridge plot stacking effect
    for i, offsetgroup in enumerate(reversed(category_order)):
        if offsetgroup not in plot_df_dict:
            continue

        category_df = plot_df_dict[offsetgroup]

        if len(category_df) == 0:
            continue

        # Get bin boundaries and densities
        lower_bounds = category_df["lower_bound"].to_list()
        upper_bounds = category_df["upper_bound"].to_list()
        densities = category_df["y"].to_list()

        # Calculate vertical offset (stack from bottom to top)
        # Reverse the offset calculation so first category is at top
        reverse_i = len(category_order) - 1 - i
        y_offset = reverse_i * vertical_spacing

        # Create stepped line for histogram
        x_vals = []
        y_vals = []
        for lower, upper, density in zip(lower_bounds, upper_bounds, densities):
            x_vals.extend([lower, upper])
            y_vals.extend([density, density])

        # Add baseline points to close the polygon
        if x_vals and y_vals:
            x_vals = [x_vals[0]] + x_vals + [x_vals[-1]]
            y_vals = [0] + y_vals + [0]
        else:
            continue

        # Apply offset to y values
        y_vals_offset = [y + y_offset for y in y_vals]

        # Determine fill color (use original order index for consistent coloring)
        original_index = category_order.index(offsetgroup)
        fill_color = (
            fill_colors[original_index % len(fill_colors)] if fill_colors else None
        )

        # Add an invisible baseline trace at the offset level
        baseline_trace = go.Scatter(
            x=x_vals,
            y=[y_offset] * len(x_vals),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
        traces.append(baseline_trace)

        # Create the ridge trace that fills from the baseline
        trace = go.Scatter(
            x=x_vals,
            y=y_vals_offset,
            mode="lines",
            name=str(offsetgroup),
            fill="tonexty",  # Fill to the previous trace (the baseline)
            fillcolor=fill_color,
            line=dict(color=line_color, width=line_width),
            showlegend=show_legend,
            hovertemplate=(
                f"<b>{offsetgroup}</b><br>"
                + "Value: %{x:.4g}<br>"
                + "Density: %{customdata:.4g}<br>"
                + "<extra></extra>"
            ),
            customdata=[y for y in y_vals],  # Store original density values
        )
        traces.append(trace)

    return traces


def ridgeplot(
    data_frame: Any,
    x: str,
    y: str,
    granularity: Optional[GranularityT] = "best",
    category_orders: Optional[dict] = None,
    labels: Optional[dict] = None,
    color_discrete_sequence: Optional[list[str]] = None,
    color_continuous_scale: Optional[str | list[str]] = None,
    opacity: float = 0.7,
    overlap: float = 0.5,
    line_width: float = 1.5,
    line_color: Optional[str] = "auto",
    log_x: bool = False,
    range_x: Optional[list] = None,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    template: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    show_legend: bool = True,
    hover_name: Optional[str] = None,
    hover_data: Optional[dict] = None,
) -> go.Figure:
    """Create a ridge plot using Khisto's optimal binning algorithm.

    A ridge plot (also known as a joy plot) displays multiple distributions
    stacked vertically with partial overlap. This visualization is particularly
    effective for comparing distributions across categories or time periods.

    This function uses Khisto's optimal binning algorithm to automatically
    determine the best bin sizes for each distribution, ensuring clear
    visualization of the underlying data patterns.

    Parameters
    ----------
    data_frame : DataFrame-like
        A DataFrame-like object (pandas, polars, etc.) containing the data.
    x : str
        Column name in `data_frame` containing the values to histogram.
    y : str
        Column name in `data_frame` containing the category labels for each
        distribution. Each unique value creates a separate ridge in the plot.
    granularity : int or 'best' or None, default 'best'
        Granularity level to use for histogram binning.
        - 'best': Uses the optimal granularity level (default)
        - int: Uses the specified granularity level
        - None: Creates an interactive slider to explore all granularity levels
    category_orders : dict, optional
        Dict mapping the category column name to an ordered list of values.
        Controls the vertical stacking order of the ridges (bottom to top).
        Example: {'category': ['Low', 'Medium', 'High']}
        If not provided, categories appear in their natural order.
    labels : dict, optional
        Dict mapping column names to custom axis/legend labels.
        Example: {'x': 'Temperature (°C)', 'y': 'Location'}
    color_discrete_sequence : list of str, optional
        List of CSS color strings to cycle through for each ridge.
        Both line and fill colors are derived from this sequence.
        Example: ['#1f77b4', '#ff7f0e', '#2ca02c']
        Ignored if `color_continuous_scale` is provided.
    color_continuous_scale : str or list of str, optional
        A continuous color scale to apply across categories in order.
        Useful for sequential data (e.g., months, years) where you want
        a gradient effect. Can be a Plotly color scale name (e.g., 'Viridis',
        'Blues', 'RdYlBu') or a list of colors.
        Example: 'Viridis', ['blue', 'yellow', 'red']
        If provided, overrides `color_discrete_sequence`.
    opacity : float, default 0.7
        Opacity of the filled areas (0 to 1). Lower values create more
        transparency, allowing overlapping ridges to remain visible.
    overlap : float, default 0.5
        Vertical overlap factor (0 to 1). Higher values create more overlap
        between adjacent distributions.
        - 0.0: No overlap (distributions stacked with gaps)
        - 0.5: Moderate overlap (default, visually pleasing)
        - 1.0: Complete overlap (distributions start at same baseline)
    line_width : float, default 1.5
        Width of the distribution outline in pixels.
    line_color : str, default 'auto'
        Color of the distribution outline. Can be:
        - 'auto': Automatically uses the plot background color (default)
        - Any CSS color string (e.g., 'white', '#FFFFFF', 'rgb(255,255,255)')
        When 'auto', the line color adapts to the template's plot background.
    log_x : bool, default False
        If True, use logarithmic scale for x-axis. Useful for data spanning
        multiple orders of magnitude.
    range_x : list, optional
        Range for x-axis as [min, max]. Restricts the visible range.
        Example: [0, 100]
    title : str, optional
        Figure title displayed at the top of the plot.
    subtitle : str, optional
        Figure subtitle displayed below the title.
    template : str, optional
        Plotly template name controlling overall styling and theme.
        Examples: 'plotly', 'plotly_white', 'plotly_dark', 'ggplot2',
        'seaborn', 'simple_white', 'presentation'
    width : int, optional
        Figure width in pixels. Example: 800
    height : int, optional
        Figure height in pixels. Example: 600
    show_legend : bool, default True
        Whether to display the legend with category names.

    Returns
    -------
    go.Figure
        A Plotly Figure object with the ridge plot visualization. The figure
        can be displayed using `fig.show()`, saved using `fig.write_html()`
        or `fig.write_image()`, or further customized using Plotly's API.

    Examples
    --------
    Create a simple ridge plot comparing distributions across categories:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from khisto.plot.plotly import ridgeplot
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'value': np.concatenate([
    ...         np.random.normal(0, 1, 500),
    ...         np.random.normal(2, 1.5, 500),
    ...         np.random.normal(-1, 0.8, 500),
    ...     ]),
    ...     'category': ['A'] * 500 + ['B'] * 500 + ['C'] * 500
    ... })
    >>>
    >>> fig = ridgeplot(df, x='value', y='category')
    >>> fig.show()

    Create a ridge plot with custom category order and colors:

    >>> fig = ridgeplot(
    ...     df,
    ...     x='value',
    ...     y='category',
    ...     category_orders={'category': ['C', 'A', 'B']},
    ...     color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],
    ...     title='Distribution Comparison'
    ... )
    >>> fig.show()

    Create a ridge plot with more overlap for a denser appearance:

    >>> fig = ridgeplot(
    ...     df,
    ...     x='value',
    ...     y='category',
    ...     overlap=0.7,
    ...     opacity=0.8,
    ...     line_width=2.0
    ... )
    >>> fig.show()

    Create a ridge plot with custom labels and styling:

    >>> fig = ridgeplot(
    ...     df,
    ...     x='value',
    ...     y='category',
    ...     labels={'value': 'Measurement', 'category': 'Group'},
    ...     template='plotly_white',
    ...     width=1000,
    ...     height=600
    ... )
    >>> fig.show()

    Create an animated ridge plot with granularity slider:

    >>> fig = ridgeplot(
    ...     df,
    ...     x='value',
    ...     y='category',
    ...     granularity=None  # Enable animation
    ... )
    >>> fig.show()

    Create a ridge plot with continuous color scale for sequential data:

    >>> # For time series or ordered categories
    >>> import plotly.express as px
    >>> fig = ridgeplot(
    ...     df_monthly,
    ...     x='temperature',
    ...     y='month',
    ...     category_orders={'month': ['Jan', 'Feb', 'Mar', ...]},
    ...     color_continuous_scale='Viridis',  # or px.colors.sequential.Viridis
    ...     title='Temperature by Month'
    ... )
    >>> fig.show()

    See Also
    --------
    khisto.plot.plotly.histogram : Create standard histograms with optimal binning
    khisto.array.histogram_series : Get full histogram information as DataFrame

    Notes
    -----
    Ridge plots are particularly effective for:

    - **Time series distributions**: Showing how a distribution evolves over time
    - **Category comparisons**: Comparing distributions across different groups
    - **Hierarchical data**: Displaying distributions at different levels

    The vertical overlap (`overlap` parameter) is key to the ridge plot aesthetic.
    The default value of 0.5 creates the characteristic "mountain range" appearance
    while maintaining readability. Experiment with this parameter to find the
    best visual balance for your data.

    Each distribution uses Khisto's optimal binning independently, so ridges with
    very different distributions can still be clearly visualized with appropriate
    bin sizes for each.

    When using the interactive granularity slider (`granularity=None`), you can
    explore how the distributions appear at different levels of detail across
    all categories simultaneously.
    """
    # Step 1: Prepare arguments and build dataframe structure
    args = locals().copy()
    granularity_param = args.pop("granularity")
    overlap_param = args.pop("overlap")
    line_width_param = args.pop("line_width")
    line_color_param = args.pop("line_color")

    # IMPORTANT: Set color=y to make Plotly create separate groups for each category
    # This is required for the grouping mechanism to work properly
    args["color"] = y
    # Initialize color_discrete_map if not provided
    if "color_discrete_map" not in args or args["color_discrete_map"] is None:
        args["color_discrete_map"] = {}

    apply_default_cascade(args)
    args = build_dataframe(args, go.Scatter)

    # Step 2: Infer plot configuration and group data
    # This will create separate groups for each category value
    trace_specs, grouped_mappings, sizeref, show_colorbar = infer_config(
        args.copy(), go.Scatter, {}, {}
    )
    grouper = [x.grouper or one_group for x in grouped_mappings] or [one_group]
    groups, orders = get_groups_and_orders(args.copy(), grouper)

    # Step 3: Compute histogram data for each group
    x_column_name = args["x"]
    histo_df_list = _compute_ridge_histogram_for_groups(
        groups, x_column_name, granularity_param, grouped_mappings
    )

    # Step 4: Ensure consistent granularity levels across groups (for animation)
    if granularity_param is None:
        histo_df_list = _fill_missing_granularities_ridge(histo_df_list)

    # Step 5: Combine all histogram data into a single DataFrame
    combined_histo_df = _combine_ridge_dataframes(histo_df_list)

    if len(combined_histo_df) == 0:
        raise ValueError("No data to plot")

    # Step 6: Determine the best granularity level to display initially
    best_granularity = _determine_best_granularity_ridge(
        combined_histo_df, granularity_param
    )

    # Step 7: Get category order
    all_categories = (
        combined_histo_df["offsetgroup"].unique(maintain_order=True).to_list()
    )

    # Apply category_orders if provided
    if category_orders and y in category_orders:
        # Use the specified order, filtering to only include categories that exist in the data
        category_order = [cat for cat in category_orders[y] if cat in all_categories]
    else:
        category_order = all_categories

    # Step 8: Set up colors
    if color_continuous_scale is not None:
        # Use continuous color scale - sample colors from the scale based on category position
        import plotly.express as px

        # Get the colorscale (either by name or use the provided list)
        if isinstance(color_continuous_scale, str):
            # It's a named colorscale
            try:
                colorscale = getattr(px.colors.sequential, color_continuous_scale)
            except AttributeError:
                try:
                    colorscale = getattr(px.colors.diverging, color_continuous_scale)
                except AttributeError:
                    # Fall back to treating it as a list
                    colorscale = [color_continuous_scale]
        else:
            colorscale = color_continuous_scale

        # Sample colors from the scale based on category index
        n_categories = len(category_order)
        if n_categories == 1:
            indices = [0]
        else:
            indices = [
                int(i * (len(colorscale) - 1) / (n_categories - 1))
                for i in range(n_categories)
            ]

        color_discrete_sequence = [colorscale[i] for i in indices]
    elif color_discrete_sequence is None:
        # Retrieve the default color sequence from Plotly Express
        import plotly.express as px

        # Get the default color sequence from the template or use Plotly's default
        if template:
            # Try to get colors from the specified template
            try:
                # Import plotly.io to access templates
                import plotly.io as pio

                # Get the template object
                template_obj = (
                    pio.templates[template]
                    if template in pio.templates
                    else pio.templates["plotly"]
                )

                # Get the colorway from the template layout (use getattr to avoid type checking issues)
                colorway = getattr(template_obj.layout, "colorway", None)
                if colorway and len(colorway) > 0:
                    color_discrete_sequence = list(colorway)
                else:
                    # Fall back to Plotly Express default
                    color_discrete_sequence = px.colors.qualitative.Plotly
            except (AttributeError, KeyError, ImportError):
                # Fall back to Plotly Express default
                color_discrete_sequence = px.colors.qualitative.Plotly
        else:
            # Use Plotly Express default color sequence
            color_discrete_sequence = px.colors.qualitative.Plotly

    # Create fill colors with opacity
    fill_colors = [
        f"rgba({int(c[1:3], 16)}, {int(c[3:5], 16)}, {int(c[5:7], 16)}, {opacity})"
        if c.startswith("#")
        else c
        for c in color_discrete_sequence
    ]

    # Step 8.5: Determine line color based on template
    if line_color_param == "auto":
        # Detect background color from template
        # Create a temporary figure with the template to extract the background color
        temp_fig = go.Figure()
        if template:
            temp_fig.update_layout(template=template)

        # Get the plot background color from the template
        try:
            plot_bgcolor = temp_fig.layout["plot_bgcolor"]
            if plot_bgcolor and plot_bgcolor != "":
                final_line_color = str(plot_bgcolor)
            else:
                # Default to white if no background color is set
                final_line_color = "white"
        except (KeyError, TypeError):
            # Default to white if we can't access the background color
            final_line_color = "white"
    else:
        final_line_color = str(line_color_param)

    # Step 9: Create traces for initial display
    traces = _create_ridge_traces(
        combined_histo_df,
        best_granularity,
        category_order,
        overlap_param,
        color_discrete_sequence,
        fill_colors,
        line_width_param,
        final_line_color,
        show_legend,
    )

    # Step 10: Calculate y-axis tick positions and labels
    # Each category is positioned at its vertical offset
    plot_df = combined_histo_df.filter(nw.col("granularity") == best_granularity)
    max_density = plot_df["y"].max()
    vertical_spacing = max_density * (1.0 - overlap_param)

    # Create tick positions - position each tick at the baseline of each ridge
    # The baseline for ridge i is at y_offset = i * vertical_spacing
    tick_positions = [i * vertical_spacing for i in range(len(category_order))]
    tick_labels = category_order

    # Step 11: Create figure layout
    x_label = labels.get(x, x) if labels else x
    y_label = labels.get(y, y) if labels else y

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_label,
            type="log" if log_x else "linear",
            range=range_x,
        ),
        yaxis=dict(
            title=y_label,
            showticklabels=True,
            tickmode="array",
            tickvals=tick_positions,
            ticktext=tick_labels,
            zeroline=False,
        ),
        template=template,
        width=width,
        height=height,
        showlegend=show_legend,
        hovermode="closest",
    )

    fig = go.Figure(data=traces, layout=layout)

    # Step 11: Add animation if granularity is None
    if granularity_param is None:
        frames = []
        max_granularity = int(combined_histo_df["granularity"].max())

        for g in range(max_granularity + 1):
            frame_traces = _create_ridge_traces(
                combined_histo_df,
                g,
                category_order,
                overlap_param,
                color_discrete_sequence,
                fill_colors,
                line_width_param,
                final_line_color,
                show_legend,
            )
            frames.append(go.Frame(data=frame_traces, name=str(g)))

        fig.frames = frames

        # Add slider
        sliders = [
            dict(
                active=best_granularity,
                yanchor="top",
                y=0,
                xanchor="left",
                x=0.1,
                currentvalue=dict(
                    prefix="Granularity: ",
                    visible=True,
                    xanchor="right",
                ),
                pad=dict(b=10, t=50),
                len=0.9,
                steps=[
                    dict(
                        args=[
                            [str(g)],
                            dict(
                                frame=dict(duration=300, redraw=True),
                                mode="immediate",
                                transition=dict(duration=300),
                            ),
                        ],
                        label=str(g),
                        method="animate",
                    )
                    for g in range(max_granularity + 1)
                ],
            )
        ]
        fig.update_layout(sliders=sliders)

    return fig

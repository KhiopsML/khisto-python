"""Matplotlib hist function combining histogram and cumulative."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union, Any, overload

import numpy as np

from khisto.utils._compat._optional import import_optional_dependency, Extras

import_optional_dependency("matplotlib", extra=Extras.MATPLOTLIB, errors="raise")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D

from .histogram import histogram as histogram_plot, _process_input_data
from .ecdf import ecdf as ecdf_plot
from khisto.array import histogram_table, ecdf_values_table

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoDataFrame, IntoSeries


@overload
def hist(
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    bins: Optional[GranularityT] = None,
    range=None,
    density: bool = False,
    weights=None,
    cumulative: bool = False,
    bottom=None,
    histtype: str = "bar",
    align: str = "mid",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    rwidth=None,
    log: bool = False,
    color: Optional[Union[str, list[str]]] = None,
    label: Optional[str] = None,
    stacked: bool = False,
    *,
    data: Optional[IntoDataFrame] = None,
    hue: None = None,
    granularity: Optional[GranularityT] = "best",
    palette: Optional[Union[str, list[str]]] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> tuple[Any, Any, Union[BarContainer, Line2D, None]]: ...


@overload
def hist(
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    bins: Optional[GranularityT] = None,
    range=None,
    density: bool = False,
    weights=None,
    cumulative: bool = False,
    bottom=None,
    histtype: str = "bar",
    align: str = "mid",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    rwidth=None,
    log: bool = False,
    color: Optional[Union[str, list[str]]] = None,
    label: Optional[str] = None,
    stacked: bool = False,
    *,
    data: Optional[IntoDataFrame] = None,
    hue: str,
    granularity: Optional[GranularityT] = "best",
    palette: Optional[Union[str, list[str]]] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Union[BarContainer, list[BarContainer], list[Line2D], None]: ...


def hist(
    x: Optional[Union[str, ArrayT, IntoSeries]] = None,
    bins: Optional[GranularityT] = None,
    range=None,
    density: bool = False,
    weights=None,
    cumulative: bool = False,
    bottom=None,
    histtype: str = "bar",
    align: str = "mid",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    rwidth=None,
    log: bool = False,
    color: Optional[Union[str, list[str]]] = None,
    label: Optional[str] = None,
    stacked: bool = False,
    *,
    data: Optional[IntoDataFrame] = None,
    hue: Optional[str] = None,
    granularity: Optional[GranularityT] = "best",
    palette: Optional[Union[str, list[str]]] = None,
    ax: Optional[Axes] = None,
    **kwargs: Any,
) -> Union[
    tuple[
        Any,
        Any,
        Union[BarContainer, list[BarContainer], Line2D, list[Line2D], None],
    ],
    Union[BarContainer, list[BarContainer], Line2D, list[Line2D], None],
]:
    """
    Compute and plot a histogram or cumulative distribution.

    This function combines the functionality of `khisto.matplotlib.histogram` and
    `khisto.matplotlib.cumulative` to provide a familiar interface similar to
    `matplotlib.pyplot.hist`, but using Khisto's optimal binning algorithm.

    Parameters
    ----------
    x : str or ArrayT or IntoSeries, optional
        Input data or column name.
    bins : int or 'best' or None, optional
        Alias for `granularity`. If provided, overrides `granularity`.
    range : tuple or None, optional
        Ignored. Khisto automatically determines the range.
    density : bool, default False
        If True, draw and return a probability density: each bin will display the
        bin's raw count divided by the total number of counts and the bin width.
    weights : array-like, optional
        Not supported yet.
    cumulative : bool, default False
        If True, then a histogram is computed where each bin gives the counts in
        that bin plus all bins for smaller values. The last bin gives the total
        number of data points.
    bottom : array-like, scalar, or None
        Not supported.
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, default 'bar'
        The type of histogram to draw.
        - 'bar' is a traditional bar-type histogram.
        - 'step' generates a lineplot (used for cumulative).
    align : {'left', 'mid', 'right'}, default 'mid'
        Ignored. Bars are always centered on the bin interval.
    orientation : {'vertical', 'horizontal'}, default 'vertical'
        If 'horizontal', barh will be used for bar-type histograms.
    rwidth : float or None, default None
        Ignored.
    log : bool, default False
        If True, the histogram axis will be set to a log scale.
    color : color or array-like of colors or None, default None
        Color spec or sequence of color specs.
    label : str or None, default None
        String, or sequence of strings to match multiple datasets.
    stacked : bool, default False
        If True, multiple data are stacked on top of each other.
        Not fully supported (use hue for grouping).
    data : IntoDataFrame, optional
        If given, `x` can be a string column name.
    hue : str, optional
        Column name for grouping.
    granularity : int or 'best' or None, default 'best'
        Granularity level for optimal binning.
    palette : str or list, optional
        Colors for hue groups.
    ax : Axes, optional
        Axes to plot on.
    **kwargs :
        kwargs are passed to the underlying plotting function.

    Returns
    -------
    n : array or list of arrays
        The values of the histogram bins.
    bins : array
        The edges of the bins.
    patches : BarContainer or Line2D or list thereof
        The return value from the plotting function.
    """
    # Handle bins alias
    if bins is not None:
        granularity = bins

    # Handle log scale
    if ax is None:
        ax = plt.gca()

    if log:
        if orientation == "vertical":
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")

    # Plotting
    if cumulative:
        # Map histtype to cumulative style
        if histtype in ["step", "stepfilled"]:
            if "drawstyle" not in kwargs:
                kwargs["drawstyle"] = "steps-post"

        patches = ecdf_plot(
            data=data,
            x=x,
            hue=hue,
            ax=ax,
            orientation=orientation,
            granularity=granularity,
            density=density,
            color=color,
            palette=palette,
            label=label,
            **kwargs,
        )
    else:
        patches = histogram_plot(
            data=data,
            x=x,
            hue=hue,
            ax=ax,
            orientation=orientation,
            granularity=granularity,
            density=density,
            color=color,
            palette=palette,
            label=label,
            **kwargs,
        )

    # If hue is None, return (n, bins, patches) to match matplotlib.pyplot.hist
    if hue is None:
        try:
            df_processed, x_col = _process_input_data(data, x)

            if cumulative:
                # For cumulative, n is the cumulative values, bins are positions
                cdf_df = ecdf_values_table(df_processed[x_col], granularity=granularity)
                value_col = (
                    "cumulative_probability" if density else "cumulative_frequency"
                )
                # Skip the first value (0) to match matplotlib behavior (values at upper edges)
                n = np.array(cdf_df[value_col].to_list()[1:])
                out_bins = np.array(cdf_df["position"].to_list())
            else:
                # For histogram, n is density/count, bins are edges
                histo_df = histogram_table(df_processed[x_col], granularity=granularity)
                value_col = "density" if density else "frequency"
                n = np.array(histo_df[value_col].to_list())

                lower = histo_df["lower_bound"].to_list()
                upper = histo_df["upper_bound"].to_list()
                if len(upper) > 0:
                    out_bins = np.array(lower + [upper[-1]])
                else:
                    out_bins = np.array([])

            return n, out_bins, patches
        except Exception:
            # Fallback if data processing fails (e.g. complex input)
            return patches

    return patches

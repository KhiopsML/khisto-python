"""Matplotlib hist function for optimal histograms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Any

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from khisto.array import histogram as khisto_histogram

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def hist(
    x: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
    cumulative: bool = False,
    histtype: str = "bar",
    orientation: Literal["vertical", "horizontal"] = "vertical",
    log: bool = False,
    color: Optional[str] = None,
    label: Optional[str] = None,
    *,
    ax: Optional[Axes] = None,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    alpha: Optional[float] = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Compute and plot an optimal histogram.

    Drop-in replacement for `matplotlib.pyplot.hist` using Khisto's optimal
    binning algorithm. Most parameters match matplotlib's hist function.

    Parameters
    ----------
    x : array_like
        Input data.
    range : tuple of (float, float), optional
        The lower and upper range of the bins. Values outside the range are
        ignored. If not provided, the range is ``(x.min(), x.max())``.
    max_bins : int, optional
        Maximum number of bins. If None, uses optimal binning.
    ax : Axes, optional
        Axes to plot on. If None, uses current axes.
    **kwargs
        Other parameters are passed to matplotlib. See `matplotlib.pyplot.hist`.

    Returns
    -------
    n : ndarray
        Histogram values.
    bins : ndarray
        Bin edges.
    patches
        Container with the bar patches.

    See Also
    --------
    matplotlib.pyplot.hist : Full documentation of supported parameters.
    khisto.histogram : Underlying histogram computation.

    Examples
    --------
    >>> from khisto.matplotlib import hist
    >>> hist(data)
    >>> hist(data, density=True, max_bins=10)
    """
    if cumulative:
        raise NotImplementedError("Cumulative histograms are not yet supported")

    # Get axes
    if ax is None:
        ax = plt.gca()

    # Compute histogram using khisto
    hist_values, bin_edges = khisto_histogram(
        x, range=range, max_bins=max_bins, density=density
    )

    # Handle log scale
    if log:
        if orientation == "vertical":
            ax.set_yscale("log")
        else:
            ax.set_xscale("log")

    # Build kwargs for plotting
    bar_kwargs: dict[str, Any] = {}
    if color is not None:
        bar_kwargs["color"] = color
    if edgecolor is not None:
        bar_kwargs["edgecolor"] = edgecolor
    if linewidth is not None:
        bar_kwargs["linewidth"] = linewidth
    if alpha is not None:
        bar_kwargs["alpha"] = alpha
    if label is not None:
        bar_kwargs["label"] = label
    bar_kwargs.update(kwargs)

    # Compute bin widths and centers
    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_widths / 2

    # Plot based on histtype
    if histtype == "bar":
        if orientation == "vertical":
            patches = ax.bar(
                bin_centers,
                hist_values,
                width=bin_widths,
                align="center",
                **bar_kwargs,
            )
        else:  # horizontal
            patches = ax.barh(
                bin_centers,
                hist_values,
                height=bin_widths,
                align="center",
                **bar_kwargs,
            )
    elif histtype in ("step", "stepfilled"):
        # Create step plot data
        step_x = np.repeat(bin_edges, 2)[1:-1]
        step_y = np.repeat(hist_values, 2)

        if histtype == "stepfilled":
            bar_kwargs.setdefault("alpha", 0.5)
            if orientation == "vertical":
                patches = ax.fill_between(step_x, step_y, step=None, **bar_kwargs)
            else:
                patches = ax.fill_betweenx(step_x, step_y, step=None, **bar_kwargs)
        else:
            if orientation == "vertical":
                patches = ax.plot(step_x, step_y, **bar_kwargs)[0]
            else:
                patches = ax.plot(step_y, step_x, **bar_kwargs)[0]
    else:
        raise ValueError(f"Unknown histtype: {histtype}")

    return hist_values, bin_edges, patches

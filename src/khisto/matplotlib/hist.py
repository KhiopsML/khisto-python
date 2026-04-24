# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Matplotlib hist function for optimal histograms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from khisto.array import histogram as khisto_histogram

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


def _normalize_cumulative(cumulative: bool | float) -> int:
    """Normalize cumulative mode to 0 (disabled), 1, or -1 (reverse)."""
    if isinstance(cumulative, (bool, np.bool_)):
        return 1 if cumulative else 0
    if isinstance(cumulative, (int, float, np.number)):
        if cumulative < 0:
            return -1
        if cumulative > 0:
            return 1
        return 0
    raise TypeError("cumulative must be a boolean or a number")


def _apply_cumulative(
    hist_values: NDArray[np.float64],
    bin_edges: NDArray[np.float64],
    *,
    density: bool,
    reverse: bool = False,
) -> NDArray[np.float64]:
    """Accumulate histogram values using matplotlib-compatible semantics."""
    if density:
        source_values = hist_values * np.diff(bin_edges)
    else:
        source_values = hist_values

    if reverse:
        return np.cumsum(source_values[::-1])[::-1]
    return np.cumsum(source_values)


def hist(
    x: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
    cumulative: bool | float = False,
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

    Parameters
    ----------
    x : array_like
        Input data. Must be 1-dimensional.
    range : tuple of (float, float), optional
        Lower and upper range of the bins. Values outside the range are
        ignored.
    max_bins : int, optional
        Maximum number of bins. If not provided, the algorithm selects
        the optimal number of bins automatically.
    density : bool, optional
        If True, return and plot probability density values. If False,
        return and plot counts. Default is False.
    cumulative : bool or float, optional
        If True, return and plot cumulative values. If negative, accumulate
        in reverse order. When used with ``density=True``, the returned values
        are cumulative probabilities so that the last (or first, in reverse)
        bin equals 1.

    Returns
    -------
    n : ndarray
        Histogram values (counts by default, or cumulative values when requested).
    bins : ndarray
        Bin edges.
    patches
        Container with the bar patches.

    See Also
    --------
    matplotlib.pyplot.hist : Matplotlib's histogram function. The ``bins``,
        ``weights``, and stacked/multiple dataset features are not supported.
    khisto.array.histogram : Underlying histogram computation.
    """
    unsupported_kwargs = {
        "bins": "Use max_bins to limit the number of bins.",
        "stacked": "Stacked histograms are not supported.",
        "weights": "Weighted histograms are not supported.",
    }
    for name, hint in unsupported_kwargs.items():
        if name in kwargs:
            raise TypeError(f"{name} is not supported. {hint}")

    cumulative_mode = _normalize_cumulative(cumulative)

    # Get axes
    if ax is None:
        ax = plt.gca()

    # Compute histogram using khisto
    hist_values, bin_edges = khisto_histogram(
        x, range=range, max_bins=max_bins, density=density
    )
    if cumulative_mode != 0:
        hist_values = _apply_cumulative(
            hist_values,
            bin_edges,
            density=density,
            reverse=cumulative_mode < 0,
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

# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Matplotlib hist function for optimal histograms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from khisto.histogram import histogram as khisto_histogram

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.patches import Polygon
    from numpy.typing import ArrayLike, NDArray


def hist(
    x: ArrayLike,
    range: tuple[float, float] | None = None,
    max_bins: int | None = None,
    density: bool = True,
    *,
    ax: Axes | None = None,
    **kwargs: Any,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    BarContainer | list[Polygon],
]:
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
        If True, returns and plots a probability density; otherwise, counts.
        Default is True.

        With adaptive binning, bin widths vary, so density and frequency
        histograms differ visually. Therefore, density is the default,
        unlike in matplotlib.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If not provided, the current axes will be used.
    **kwargs :
        other keyword arguments are described in ``matplotlib.pyplot.hist``. The ``bins``,
        ``weights``, ``stacked``, ``histtype="barstacked"``, and multiple dataset
        features are not supported.

    Returns
    -------
    n : ndarray
        Histogram values (counts by default, or cumulative values when requested).
    bins : ndarray
        Bin edges.
    patches
        Container with the bar patches, or a list containing the step polygon.

    .. note::
        Khiops bins are left-open and right-closed, ``(lower, upper]``, unlike
        Matplotlib bins which are left-closed and right-open, ``[lower, upper)``.
        The displayed plot is still correct.

    See Also
    --------
    matplotlib.pyplot.hist : Matplotlib's histogram function.
    khisto.histogram : Underlying histogram computation.
    """
    # optional dependency; only import if strictly needed.
    import matplotlib.pyplot as plt
    from matplotlib.container import BarContainer
    from matplotlib.patches import Polygon

    unsupported_kwargs = {
        "bins": "Use max_bins to limit the number of bins.",
        "stacked": "Stacked histograms are not supported.",
        "weights": "Weighted histograms are not supported.",
    }
    for name, hint in unsupported_kwargs.items():
        if name in kwargs:
            raise TypeError(f"{name} is not supported. {hint}")

    histtype = kwargs.get("histtype", "bar")
    if histtype == "barstacked":
        raise ValueError(
            "histtype='barstacked' is not supported. Khisto only accepts a single dataset."
        )

    # Use frequencies so Matplotlib applies density and cumulative only once.
    frequencies, bin_edges = khisto_histogram(
        x,
        range=range,
        max_bins=max_bins,
        density=False,
    )

    if ax is None:
        ax = plt.gca()

    # Weighted left edges preserve Khiops' right-closed bins and [-1e100, 1e100]
    # clamping when Matplotlib renders its left-closed bins.
    values, edges, patches = ax.hist(
        bin_edges[:-1].tolist(),
        bin_edges.tolist(),
        weights=frequencies.tolist(),
        density=density,
        range=range,
        **kwargs,
    )
    if isinstance(values, list):
        raise TypeError("Matplotlib unexpectedly returned multiple histograms.")
    if isinstance(patches, BarContainer):
        histogram_patches: BarContainer | list[Polygon] = patches
    elif isinstance(patches, list):
        histogram_patches = [patch for patch in patches if isinstance(patch, Polygon)]
        if len(histogram_patches) != len(patches):
            raise TypeError("Matplotlib returned unexpected histogram patches.")
    else:
        raise TypeError("Matplotlib returned unexpected histogram patches.")

    if histtype == "bar" and "edgecolor" not in kwargs:
        if not isinstance(histogram_patches, BarContainer):
            raise TypeError("Matplotlib unexpectedly returned non-bar patches.")
        for patch in histogram_patches.patches:
            patch.set_edgecolor(patch.get_facecolor())

    return values, edges, histogram_patches

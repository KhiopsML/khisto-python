# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Matplotlib hist function for optimal histograms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from matplotlib.axes import Axes

from khisto.array import histogram as khisto_histogram

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def hist(
    x: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = True,
    *,
    ax: Optional[Axes] = None,
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
        If True, returns and plots a probability density; otherwise, counts.
        Default is True.

        With adaptive binning, bin widths vary, so density and frequency
        histograms differ visually. Therefore, density is the default,
        unlike in matplotlib.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If not provided, the current axes will be used.
    **kwargs :
        other keyword arguments are described in ``matplotlib.pyplot.hist``. The ``bins``,
        ``weights``, and stacked/multiple dataset features are not supported.

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
    matplotlib.pyplot.hist : Matplotlib's histogram function.
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

    # Compute histogram using khisto
    _, bin_edges = khisto_histogram(x, range=range, max_bins=max_bins, density=density)

    if ax is None:
        # optional dependency; only import if strictly needed.
        import matplotlib.pyplot as plt

        ax = plt.gca()

    return ax.hist(x, bin_edges, density=density, range=range, **kwargs)

# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Matplotlib hist function for optimal histograms."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from khisto.histogram import histogram as khisto_histogram

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.patches import Polygon
    from numpy.typing import ArrayLike, NDArray


@overload
def hist(
    x: ArrayLike,
    range: tuple[float, float] | None = None,
    max_bins: int | None = None,
    density: bool = True,
    *,
    ax: Axes | None = None,
    histtype: Literal["bar"] = "bar",
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], BarContainer]: ...


@overload
def hist(
    x: ArrayLike,
    range: tuple[float, float] | None = None,
    max_bins: int | None = None,
    density: bool = True,
    *,
    ax: Axes | None = None,
    histtype: Literal["step", "stepfilled"],
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[Polygon]]: ...


@overload
def hist(
    x: ArrayLike,
    range: tuple[float, float] | None = None,
    max_bins: int | None = None,
    density: bool = True,
    *,
    ax: Axes | None = None,
    histtype: str,
    **kwargs: Any,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    BarContainer | list[Polygon],
]: ...


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

        By default, bar edges use the face color's RGB values with alpha 0.5.
        An explicit ``alpha`` applies to both fill and edges. ``edgecolor``
        (or ``ec``) overrides the default edge styling.

    Returns
    -------
    n : ndarray
        Histogram values (counts by default, or cumulative values when requested).
    bins : ndarray
        Bin edges.
    patches
        Container with the bar patches, or a list containing the step polygon.

    Raises
    ------
    ImportError
        If Matplotlib is not installed. Install plotting support with
        ``pip install "khisto[matplotlib]"``.

    .. note::
        Khiops bins are left-open and right-closed, ``(lower, upper]``, unlike
        Matplotlib bins which are left-closed and right-open, ``[lower, upper)``.
        The displayed plot is still correct.

    See Also
    --------
    matplotlib.pyplot.hist : Matplotlib's histogram function.
    khisto.histogram : Underlying histogram computation.
    """
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError(
            'Plotting requires Matplotlib. Install it with: pip install "khisto[matplotlib]"'
        )

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
    cumulative = kwargs.get("cumulative", False)
    plot_weights = (
        frequencies / frequencies.sum() if density and cumulative else frequencies
    )
    values, edges, patches = ax.hist(
        x=bin_edges[:-1],
        bins=bin_edges.tolist(),
        weights=plot_weights,
        density=density and not cumulative,
        **kwargs,
    )
    if isinstance(values, list):
        raise TypeError("Matplotlib unexpectedly returned multiple histograms.")
    if isinstance(patches, BarContainer):
        histogram_patches: BarContainer | list[Polygon] = patches
    elif isinstance(patches, list):
        histogram_patches = [patch for patch in patches if isinstance(patch, Polygon)]
        if len(histogram_patches) != len(patches):
            raise TypeError(
                "Matplotlib returned unexpected histogram patches. "
                "The patches should all be of type Polygon."
            )
    else:
        raise TypeError(
            "Matplotlib returned unexpected histogram patches of type "
            f"{type(patches).__name__}; expected BarContainer or list of Polygon."
        )

    # Adaptive bins can be narrower than a pixel and vanish when only filled.
    # Drawing a translucent edge in the face color keeps them visible.
    # The patches are rendered at draw time, so updating them here still applies.
    if histtype == "bar" and not {"edgecolor", "ec"}.intersection(kwargs.keys()):
        if not isinstance(histogram_patches, BarContainer):
            raise TypeError("Matplotlib unexpectedly returned non-bar patches.")
        for patch in histogram_patches.patches:
            patch.set_edgecolor((*patch.get_facecolor()[:3], 0.5))

    return values, edges, histogram_patches

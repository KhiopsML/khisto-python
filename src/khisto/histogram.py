# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Optimal histogram functions with a NumPy-compatible interface."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from khisto.core import HistogramResult, compute_histograms


def _select_histogram(
    histogram_results: list[HistogramResult],
    max_bins: int | None = None,
) -> HistogramResult:
    """Select the appropriate histogram from the list of results.

    Parameters
    ----------
    histogram_results : list[HistogramResult]
        List of histogram results at different granularity levels.
    max_bins : int, optional
        Maximum number of bins. If None, return the best (optimal) histogram.

    Returns
    -------
    HistogramResult
        The selected histogram result.
    """

    # Find the best histogram marked as is_best,
    # or default to the last one if none is marked.
    best_histogram = next(
        (result for result in reversed(histogram_results) if result.is_best),
        histogram_results[-1],
    )

    if max_bins is None:
        return best_histogram

    # Select the histogram with the highest granularity
    # that does not exceed max_bins.
    # Histograms finer than the best interpretable one are skipped.
    return next(
        (
            result
            for result in reversed(histogram_results)
            if result.granularity <= best_histogram.granularity
            and len(result) <= max_bins
        ),
        histogram_results[0],
    )


def histogram(
    a: ArrayLike,
    range: tuple[float, float] | None = None,
    max_bins: int | None = None,
    density: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute an optimal histogram using the Khiops binning algorithm.

    .. note::
       Khiops bins are right-closed, ``(lower, upper]``, unlike Numpy bins which
       are left-closed, ``[lower, upper)``.

    Parameters
    ----------
    a : array_like
        Input data. Must be 1-dimensional.
    range : tuple of (float, float), optional
        The lower and upper range of the bins. Values outside the range are
        ignored. If not provided, the range is ``(a.min(), a.max())``.
    max_bins : int, optional
        Maximum number of bins. If not provided, the algorithm selects
        the optimal number of bins automatically.
    density : bool, optional
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at the bin, normalized such that the integral over the range is 1.
        Default is True.

    Returns
    -------
    hist : ndarray
        The values of the histogram.
    bin_edges : ndarray
        The bin edges (length(hist) + 1).

    See Also
    --------
    numpy.histogram : NumPy's histogram function (``bins`` and ``weights``
        parameters are not supported).

    Notes
    -----
    Unlike numpy.histogram, this function uses optimal binning which may
    produce bins of unequal width. The bins are determined by the Khiops
    algorithm to best represent the underlying data distribution.

    The method implemented in Khiops is comprehensively detailed in [2]_ and
    further extended in [1]_.

    References
    ----------
    .. [1] M. Boulle. Floating-point histograms for exploratory analysis of
       large scale real-world data sets. Intelligent Data Analysis,
       28(5):1347-1394, 2024.
    .. [2] V. Zelaya Mendizabal, M. Boulle, F. Rossi. Fast and fully-automated
       histograms for large-scale data sets. Computational Statistics & Data
       Analysis, 180:0-0, 2023.
    """
    array = np.asarray(a, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError(
            f"Expected 1-D array, got {array.ndim}-D array instead. "
            "Reshape your data or flatten it before calling histogram."
        )

    if max_bins is not None and max_bins <= 0:
        raise ValueError("max_bins must be a positive integer or None.")

    # Filter values by range if specified
    if range is not None:
        min_value, max_value = range
        array = array[(array >= min_value) & (array <= max_value)]

    histogram_results = compute_histograms(array)
    histogram_result = _select_histogram(histogram_results, max_bins=max_bins)

    if density:
        return histogram_result.densities.copy(), histogram_result.bin_edges.copy()
    return (
        histogram_result.frequencies.astype(np.float64),
        histogram_result.bin_edges.copy(),
    )

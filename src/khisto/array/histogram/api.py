# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Optimal histogram functions with numpy-compatible interface."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from khisto.core import HistogramResult, compute_histograms


def _select_histogram(
    histogram_results: list[HistogramResult],
    max_bins: Optional[int] = None,
) -> HistogramResult:
    """Select the appropriate histogram from the list of results.

    Parameters
    ----------
    results : list[HistogramResult]
        List of histogram results at different granularity levels.
    max_bins : int, optional
        Maximum number of bins. If None, return the best (optimal) histogram.

    Returns
    -------
    HistogramResult
        The selected histogram result.
    """
    if max_bins is not None:
        # Find the finest granularity that respects max_bins
        selected = None
        for r in histogram_results:
            if len(r) <= max_bins:
                selected = r
            else:
                break
        # If no histogram respects the constraint, use the coarsest one
        return selected if selected is not None else histogram_results[0]
    else:
        # Return the best (optimal) histogram
        for r in reversed(histogram_results):
            if r.is_best:
                return r
        # Fallback to finest granularity if no best is marked
        return histogram_results[-1]


def histogram(
    a: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute an optimal histogram using the Khiops binning algorithm.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
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
        Default is False.

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
    arr = np.asarray(a, dtype=np.float64).flatten()

    if max_bins is not None and max_bins <= 0:
        raise ValueError("max_bins must be a positive integer or None.")

    # Filter values by range if specified
    if range is not None:
        min_val, max_val = range
        arr = arr[(arr >= min_val) & (arr <= max_val)]

    histogram_results = compute_histograms(arr)
    histogram_result = _select_histogram(histogram_results, max_bins=max_bins)

    if density:
        return histogram_result.density.copy(), histogram_result.bin_edges.copy()
    else:
        return histogram_result.frequency.astype(
            np.float64
        ), histogram_result.bin_edges.copy()

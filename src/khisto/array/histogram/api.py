# Copyright (c) 2026-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Optimal histogram functions with numpy-compatible interface.

This module provides a numpy.histogram-like interface for computing
optimal histograms using the Khiops binning algorithm.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from khisto.core import compute_histogram, HistogramResult


def _select_histogram(
    results: list[HistogramResult],
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
        for r in results:
            if len(r) <= max_bins:
                selected = r
            else:
                break
        # If no histogram respects the constraint, use the coarsest one
        return selected if selected is not None else results[0]
    else:
        # Return the best (optimal) histogram
        for r in results:
            if r.is_best:
                return r
        # Fallback to finest granularity if no best is marked
        return results[-1]


def histogram(
    a: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: Optional[bool] = None,
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
        Default is None, which behaves as False for backwards compatibility.

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
    """
    # Convert to numpy array and flatten
    arr = np.asarray(a, dtype=np.float64).ravel()

    # Filter values by range if specified
    if range is not None:
        min_val, max_val = range
        arr = arr[(arr >= min_val) & (arr <= max_val)]

    results = compute_histogram(arr)
    result = _select_histogram(results, max_bins=max_bins)

    # Treat None as False for backwards compatibility (like NumPy)
    if density:
        return result.density.copy(), result.bin_edges.copy()
    else:
        return result.frequency.astype(np.float64), result.bin_edges.copy()

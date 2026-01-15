"""Optimal histogram functions with numpy-compatible interface.

This module provides a numpy.histogram-like interface for computing
optimal histograms using the Khiops binning algorithm.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from khisto.core import compute_histogram, HistogramResult


def histogram(
    a: ArrayLike,
    range: Optional[tuple[float, float]] = None,
    max_bins: Optional[int] = None,
    density: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute an optimal histogram using the Khiops binning algorithm.

    This function is a drop-in replacement for numpy.histogram, but uses
    optimal binning instead of equal-width bins.

    Parameters
    ----------
    a : array_like
        Input data. The histogram is computed over the flattened array.
    range : tuple of (float, float), optional
        The lower and upper range of the bins. If not provided, range is
        simply (a.min(), a.max()). Values outside the range are ignored.
    max_bins : int, optional
        Maximum number of bins. If not provided, the algorithm selects
        the optimal number of bins automatically.
    density : bool, default False
        If False, the result will contain the number of samples in each bin.
        If True, the result is the value of the probability density function
        at each bin, normalized such that the integral over the range is 1.

    Returns
    -------
    hist : ndarray
        The values of the histogram. If density is True, these are
        probability density values; otherwise, they are counts.
    bin_edges : ndarray
        The bin edges (length(hist) + 1).

    See Also
    --------
    numpy.histogram : NumPy's standard histogram function.

    Notes
    -----
    Unlike numpy.histogram, this function uses optimal binning which may
    produce bins of unequal width. The bins are determined by the Khiops
    algorithm to best represent the underlying data distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import histogram
    >>> data = np.random.normal(0, 1, 1000)
    >>> hist, bin_edges = histogram(data)
    >>> # Density histogram
    >>> density_hist, edges = histogram(data, density=True)
    >>> # Constrained number of bins
    >>> hist, edges = histogram(data, max_bins=10)
    >>> # Limited range
    >>> hist, edges = histogram(data, range=(-2, 2))
    """
    # Convert to numpy array and flatten
    arr = np.asarray(a, dtype=np.float64).ravel()

    result = compute_histogram(arr, max_bins=max_bins, range=range)
    assert isinstance(result, HistogramResult)  # return_all defaults to False

    if density:
        return result.density.copy(), result.bin_edges.copy()
    else:
        return result.frequency.astype(np.float64), result.bin_edges.copy()

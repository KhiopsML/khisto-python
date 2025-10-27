"""Density histogram functions using optimal binning.

This module provides functions to compute density-based histograms using the
Khiops optimal binning algorithm. For cumulative distribution functions (CDF),
see :mod:`khisto.array.cumulative`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import narwhals as nw

from khisto.core import compute_histogram

from ._shared import (
    build_edge_positions,
    extract_bin_edges,
    prepare_input,
    prepare_input_for_df,
    validate_granularity,
)

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoSeries


def histogram(
    x: Union[ArrayT, IntoSeries],
    granularity: GranularityT = "best",
) -> tuple[ArrayT, ArrayT]:
    """Compute density histogram using optimal binning.

    This function computes a density-based histogram. The bins are determined using
    the Khiops optimal binning algorithm which adapts to the data distribution.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data. Accepts any Python Array API compliant array, list/tuple,
        array.array, or a Narwhals Series.
    granularity : int or "best", default "best"
        Granularity level to use. "best" selects the heuristic best histogram.
        If an integer, uses that granularity level (capped to the maximum available).

    Returns
    -------
    densities : ArrayT
        Density values for each bin. For unequal bin widths, the integral of
        density × width over all bins equals 1.
    bin_edges : ArrayT
        Sorted bin boundary array of length ``len(densities) + 1``.

    See Also
    --------
    histogram_df : Histogram as a DataFrame with additional metadata.
    histogram_bin_edges : Get only the bin edges.
    khisto.array.cumulative_distribution : Cumulative distribution function.

    Notes
    -----
    * This function returns density values, not counts or probabilities.
    * Bins may have unequal widths; density normalizes for bin width.
    * For cumulative distributions, use :func:`~khisto.array.cumulative_distribution`.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import histogram
    >>> data = np.random.normal(0, 1, 1000)
    >>> densities, edges = histogram(data)
    >>> # Verify normalization: sum(densities * np.diff(edges)) ≈ 1.0
    """
    arrow_array, backend = prepare_input(x)
    validate_granularity(granularity)

    df = compute_histogram(arrow_array, granularity=granularity)

    lower_bounds, last_upper_bound = extract_bin_edges(df)
    densities = df["density"].combine_chunks()
    bin_edges = build_edge_positions(lower_bounds, last_upper_bound)

    return backend.asarray(densities), backend.asarray(bin_edges)


def histogram_bin_edges(
    x: Union[ArrayT, IntoSeries], granularity: GranularityT = "best"
) -> ArrayT:
    """Compute histogram bin edges using optimal binning.

    This function returns only the bin edge positions determined by the Khiops
    optimal binning algorithm, without computing densities or probabilities.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data. Accepts any Python Array API compliant array, list/tuple,
        array.array, or a Narwhals Series.
    granularity : int or "best", default "best"
        Granularity level to use. If "best", uses the best histogram.
        If an integer, uses the histogram at that granularity level.
        If the provided granularity is higher than the most granular, uses the most granular.

    Returns
    -------
    ArrayT
        Sorted array of bin edges (lower and upper bounds).

    See Also
    --------
    histogram : Compute density histogram with bin edges.
    histogram_df : Histogram as a DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import histogram_bin_edges
    >>> data = np.random.normal(0, 1, 1000)
    >>> edges = histogram_bin_edges(data)
    >>> # Use edges with np.histogram or other tools
    """
    arrow_array, backend = prepare_input(x)
    validate_granularity(granularity)

    df = compute_histogram(arrow_array, granularity=granularity)

    lower_bounds, last_upper_bound = extract_bin_edges(df)
    bin_edges = build_edge_positions(lower_bounds, last_upper_bound)

    return backend.asarray(bin_edges)


def histogram_df(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = None,
) -> nw.DataFrame:
    """Return detailed histogram information as a DataFrame.

    This function provides comprehensive histogram metadata including bin bounds,
    frequencies, probabilities, and densities. For cumulative distribution data,
    use :func:`~khisto.array.cumulative_distribution_df`.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data. Accepts any Python Array API compliant array, list/tuple,
        array.array, or a Narwhals Series.
    granularity : int | 'best' | None, optional
        * ``None`` -> return all granularities.
        * ``'best'`` -> only best histogram.
        * integer -> that granularity level. If the provided granularity is higher
          than the most granular, uses the most granular.

    Returns
    -------
    nw.DataFrame
        Columns include:

        - ``lower_bound``: Lower edge of each bin
        - ``upper_bound``: Upper edge of each bin
        - ``length``: Number of values in the bin
        - ``frequency``: Relative frequency (length / total_count)
        - ``probability``: Bin probability (same as frequency)
        - ``density``: Probability density (frequency / bin_width)
        - ``center``: Bin center position
        - ``granularity``: Granularity level
        - ``is_best``: Boolean indicating the best histogram

    See Also
    --------
    histogram : Return arrays of densities and bin edges.
    khisto.array.cumulative_distribution_df : CDF as a DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import histogram_df
    >>> data = np.random.normal(0, 1, 1000)
    >>> df = histogram_df(data, granularity="best")
    >>> # df contains detailed bin information
    """
    arrow_array, narwhals_backend = prepare_input_for_df(x)
    table = compute_histogram(arrow_array, granularity=granularity)
    return nw.from_arrow(table, backend=narwhals_backend)

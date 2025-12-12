"""Density histogram functions using optimal binning.

This module provides functions to compute density-based histograms using the
Khiops optimal binning algorithm. For cumulative distribution functions (CDF),
see :mod:`khisto.array.cumulative`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union, cast, overload

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from khisto.core import compute_histogram

from .._shared import (
    build_edge_positions,
    extract_bin_edges,
    prepare_input,
    prepare_input_for_df,
    validate_granularity,
)

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoSeries


@overload
def histogram(
    x: Union[ArrayT, IntoSeries],
    granularity: None,
    density: bool = ...,
) -> list[tuple[ArrayT, ArrayT]]: ...


@overload
def histogram(
    x: Union[ArrayT, IntoSeries],
    granularity: Union[int, Literal["best"]] = ...,
    density: bool = ...,
) -> tuple[ArrayT, ArrayT]: ...


def histogram(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
    density: bool = True,
) -> Union[tuple[ArrayT, ArrayT], list[tuple[ArrayT, ArrayT]]]:
    """Compute histogram using optimal binning.

    This function computes a histogram using the Khiops optimal binning algorithm
    which adapts to the data distribution.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data. Accepts any Python Array API compliant array, list/tuple,
        array.array, or a Narwhals Series.
    granularity : int, "best", or None, default "best"
        Granularity level to use:

        - ``"best"``: Selects the heuristic best histogram.
        - ``int``: Uses that granularity level (capped to the maximum available).
        - ``None``: Returns histograms for all granularity levels.

    density : bool, default True
        If ``True``, return probability density values. For unequal bin widths,
        the integral of density x width over all bins equals 1.
        If ``False``, return the frequency (count) for each bin.

    Returns
    -------
    tuple[ArrayT, ArrayT] or list[tuple[ArrayT, ArrayT]]
        If ``granularity`` is ``"best"`` or an integer:
            A tuple of (values, bin_edges) where values are density or frequency
            values for each bin, and bin_edges is a sorted boundary array.
        If ``granularity`` is ``None``:
            A list of (values, bin_edges) tuples, one for each granularity level,
            sorted by granularity (coarsest to finest).

    See Also
    --------
    histogram_table : Histogram as a DataFrame with additional metadata.
    histogram_bin_edges : Get only the bin edges.
    khisto.array.ecdf : Empirical cumulative distribution function.

    Notes
    -----
    * Bins may have unequal widths determined by the optimal binning algorithm.
    * For cumulative distributions, use :func:`~khisto.array.ecdf`.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import histogram
    >>> data = np.random.normal(0, 1, 1000)
    >>> densities, edges = histogram(data)  # Default: density=True
    >>> frequencies, edges = histogram(data, density=False)
    >>> all_histograms = histogram(data, granularity=None)  # All granularities
    >>> # Verify normalization: sum(densities * np.diff(edges)) ≈ 1.0
    """
    arrow_array, backend = prepare_input(x)

    if granularity is None:
        # Return all granularities as a list of tuples
        df = compute_histogram(arrow_array, granularity=None)
        column_name = "density" if density else "frequency"

        # Get unique granularities sorted
        granularities = sorted(
            cast(list[int], pc.unique(df["granularity"]).to_pylist())
        )

        results = []
        for g in granularities:
            mask = pc.equal(df["granularity"], pa.scalar(g, type=pa.int32()))
            g_df = df.filter(mask)

            lower_bounds, last_upper_bound = extract_bin_edges(g_df)
            bin_edges = build_edge_positions(lower_bounds, last_upper_bound)
            output_column = g_df[column_name].combine_chunks()

            results.append((backend.asarray(output_column), backend.asarray(bin_edges)))

        return results

    validate_granularity(granularity)
    df = compute_histogram(arrow_array, granularity=granularity)

    lower_bounds, last_upper_bound = extract_bin_edges(df)
    bin_edges = build_edge_positions(lower_bounds, last_upper_bound)

    # Select output column based on density parameter
    column_name = "density" if density else "frequency"
    output_column = df[column_name].combine_chunks()

    print(backend, "rrrrrrrrrr")
    return backend.asarray(output_column), backend.asarray(bin_edges)


@overload
def histogram_bin_edges(
    x: Union[ArrayT, IntoSeries],
    granularity: None,
) -> list[ArrayT]: ...


@overload
def histogram_bin_edges(
    x: Union[ArrayT, IntoSeries],
    granularity: Union[int, Literal["best"]] = ...,
) -> ArrayT: ...


def histogram_bin_edges(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
) -> Union[ArrayT, list[ArrayT]]:
    """Compute histogram bin edges using optimal binning.

    This function returns only the bin edge positions determined by the Khiops
    optimal binning algorithm, without computing densities or probabilities.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data. Accepts any Python Array API compliant array, list/tuple,
        array.array, or a Narwhals Series.
    granularity : int, "best", or None, default "best"
        Granularity level to use:

        - ``"best"``: Uses the best histogram.
        - ``int``: Uses the histogram at that granularity level
          (capped to maximum available).
        - ``None``: Returns bin edges for all granularity levels.

    Returns
    -------
    ArrayT or list[ArrayT]
        If ``granularity`` is ``"best"`` or an integer:
            Sorted array of bin edges (lower and upper bounds).
        If ``granularity`` is ``None``:
            A list of bin edge arrays, one for each granularity level,
            sorted by granularity (coarsest to finest).

    See Also
    --------
    histogram : Compute density histogram with bin edges.
    histogram_table : Histogram as a DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import histogram_bin_edges
    >>> data = np.random.normal(0, 1, 1000)
    >>> edges = histogram_bin_edges(data)
    >>> all_edges = histogram_bin_edges(data, granularity=None)  # All granularities
    >>> # Use edges with np.histogram or other tools
    """
    arrow_array, backend = prepare_input(x)

    df = compute_histogram(arrow_array, granularity=granularity)

    if granularity is None:
        # Get unique granularities sorted
        granularities = sorted(
            cast(list[int], pc.unique(df["granularity"]).to_pylist())
        )

        results = []
        for g in granularities:
            mask = pc.equal(df["granularity"], pa.scalar(g, type=pa.int32()))
            g_df = df.filter(mask)

            lower_bounds, last_upper_bound = extract_bin_edges(g_df)
            bin_edges = build_edge_positions(lower_bounds, last_upper_bound)

            results.append(backend.asarray(bin_edges))

        return results

    lower_bounds, last_upper_bound = extract_bin_edges(df)
    bin_edges = build_edge_positions(lower_bounds, last_upper_bound)

    return backend.asarray(bin_edges)


def histogram_table(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = None,
) -> nw.DataFrame:
    """Return detailed histogram information as a DataFrame.

    This function provides comprehensive histogram metadata including bin bounds,
    frequencies, probabilities, and densities. For cumulative distribution data,
    use :func:`~khisto.array.ecdf_values_table`.

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
    khisto.array.ecdf_values_table : ECDF values as a DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import histogram_table
    >>> data = np.random.normal(0, 1, 1000)
    >>> df = histogram_table(data, granularity="best")
    >>> # df contains detailed bin information
    """
    arrow_array, narwhals_backend = prepare_input_for_df(x)
    table = compute_histogram(arrow_array, granularity=granularity)
    return nw.from_arrow(table, backend=narwhals_backend)

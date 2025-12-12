"""Cumulative distribution functions for optimal histograms.

This module provides functions to compute cumulative distribution functions (CDF)
from optimal histogram bins. The CDF functions are separated from density-based
histogram functions to maintain clear API semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union, cast, overload

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from khisto.core import compute_histogram

from .._shared import prepare_input, validate_granularity
from .core import ECDFResult, ECDFResultCollection
from .utils import (
    _compute_cdf_positions,
    _compute_cdf_values,
    _create_granularity_cdf_df,
)

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoSeries


# ============================================================================
# Public API: ecdf (returns callable ECDF object)
# ============================================================================


@overload
def ecdf(
    x: Union[ArrayT, IntoSeries],
    granularity: None,
) -> ECDFResultCollection: ...


@overload
def ecdf(
    x: Union[ArrayT, IntoSeries],
    granularity: Union[int, Literal["best"]] = ...,
) -> ECDFResult: ...


def ecdf(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
) -> Union[ECDFResult, ECDFResultCollection]:
    """Compute an empirical CDF that can be evaluated at any point.

    Returns an ECDF object (or collection of objects) that supports evaluation
    at arbitrary points using linear interpolation between bin edges.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data. Accepts any Python Array API compliant array, list/tuple,
        array.array, or a Narwhals Series.
    granularity : int, "best", or None, default "best"
        Granularity level to use:

        - ``"best"``: Returns ECDF for the heuristic best histogram.
        - ``int``: Uses that granularity level (capped to the maximum available).
        - ``None``: Returns an ECDFCollection with all granularity levels.

    Returns
    -------
    ECDF or ECDFCollection
        If ``granularity`` is ``"best"`` or an integer:
            An ECDF object that can be called to evaluate the CDF at any point.
        If ``granularity`` is ``None``:
            An ECDFCollection containing ECDFs for all granularity levels.

    See Also
    --------
    ecdf_values : Get discrete CDF values at bin edges as arrays.
    ecdf_values_table : Get CDF values as a DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import ecdf
    >>> data = np.random.normal(0, 1, 1000)

    >>> # Get ECDF for best granularity
    >>> cdf_func = ecdf(data)
    >>> cdf_func(0.0)  # Evaluate at x=0
    >>> cdf_func(np.linspace(-3, 3, 100))  # Evaluate at many points

    >>> # Get ECDFs for all granularities
    >>> cdf_collection = ecdf(data, granularity=None)
    >>> cdf_collection.best(0.0)  # Use best granularity
    >>> cdf_collection[2](0.0)  # Use granularity level 2
    """
    arrow_array, backend = prepare_input(x)

    validate_granularity(granularity)
    df = compute_histogram(arrow_array, granularity=granularity)

    if granularity is None:
        # Get unique granularities sorted
        granularities = sorted(
            cast(list[int], pc.unique(df["granularity"]).to_pylist())
        )

        # Find best granularity
        best_mask = pc.equal(df["is_best"], pa.scalar(True))
        best_df = df.filter(best_mask)
        best_granularity = (
            best_df["granularity"][0].as_py() if len(best_df) > 0 else granularities[-1]
        )

        ecdfs = []
        for g in granularities:
            mask = pc.equal(df["granularity"], pa.scalar(g, type=pa.int32()))
            g_df = df.filter(mask)

            cdf_vals = _compute_cdf_values(g_df, density=True)
            positions = _compute_cdf_positions(g_df)

            ecdfs.append(
                ECDFResult(
                    positions=backend.asarray(positions),
                    cdf_values=backend.asarray(cdf_vals),
                    granularity=g,
                    is_best=(g == best_granularity),
                )
            )

        return ECDFResultCollection(ecdfs)

    # Single granularity
    cdf_vals = _compute_cdf_values(df, density=True)
    positions = _compute_cdf_positions(df)

    # Check if this is best
    is_best = df["is_best"][0].as_py() if "is_best" in df.column_names else True
    gran = df["granularity"][0].as_py() if "granularity" in df.column_names else 0

    return ECDFResult(
        positions=backend.asarray(positions),
        cdf_values=backend.asarray(cdf_vals),
        granularity=gran,
        is_best=is_best,
    )


# ============================================================================
# Public API: ecdf_values (returns discrete arrays)
# ============================================================================


@overload
def ecdf_values(
    x: Union[ArrayT, IntoSeries],
    granularity: None,
    density: bool = ...,
) -> list[tuple[ArrayT, ArrayT]]: ...


@overload
def ecdf_values(
    x: Union[ArrayT, IntoSeries],
    granularity: Union[int, Literal["best"]] = ...,
    density: bool = ...,
) -> tuple[ArrayT, ArrayT]: ...


def ecdf_values(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
    density: bool = True,
) -> Union[tuple[ArrayT, ArrayT], list[tuple[ArrayT, ArrayT]]]:
    """Compute discrete ECDF values at bin edges.

    Returns the discrete sample points of the empirical CDF aligned with bin edges.
    The first value is 0.0 and the last value is 1.0 for density=True,
    or 0 and the total count for density=False.

    For a callable ECDF that supports evaluation at arbitrary points,
    use :func:`ecdf` instead.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data. Accepts any Python Array API compliant array, list/tuple,
        array.array, or a Narwhals Series.
    granularity : int, "best", or None, default "best"
        Granularity level to use:

        - ``"best"``: Selects the heuristic best histogram.
        - ``int``: Uses that granularity level (capped to the maximum available).
        - ``None``: Returns values for all granularity levels.

    density : bool, default True
        If ``True``, return cumulative probability values ranging from 0.0 to 1.0.
        If ``False``, return cumulative frequency counts ranging from 0 to total count.

    Returns
    -------
    tuple[ArrayT, ArrayT] or list[tuple[ArrayT, ArrayT]]
        If ``granularity`` is ``"best"`` or an integer:
            A tuple of (cdf_values, positions) where cdf_values contains cumulative
            values at each bin edge, and positions are the corresponding edge positions.
        If ``granularity`` is ``None``:
            A list of (cdf_values, positions) tuples, one for each granularity level,
            sorted by granularity (coarsest to finest).

    See Also
    --------
    ecdf : Get a callable ECDF object for evaluation at arbitrary points.
    ecdf_values_table : Get CDF values as a DataFrame with additional metadata.
    khisto.array.histogram : Compute density histogram.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import ecdf_values
    >>> data = np.random.normal(0, 1, 1000)
    >>> cdf_vals, edges = ecdf_values(data)  # Default: density=True
    >>> cdf_freq, edges = ecdf_values(data, density=False)
    >>> all_cdfs = ecdf_values(data, granularity=None)  # All granularities
    >>> # cdf_vals[0] = 0.0, cdf_vals[-1] = 1.0 (for density=True)
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

            cdf_vals = _compute_cdf_values(g_df, density)
            positions = _compute_cdf_positions(g_df)

            results.append((backend.asarray(cdf_vals), backend.asarray(positions)))

        return results

    cdf_vals = _compute_cdf_values(df, density)
    positions = _compute_cdf_positions(df)

    return backend.asarray(cdf_vals), backend.asarray(positions)


# ============================================================================
# Public API: ecdf_values_table (returns DataFrame)
# ============================================================================


def ecdf_values_table(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = None,
) -> nw.DataFrame:
    """Return ECDF values as a DataFrame.

    This function computes the empirical CDF from the optimal histogram bins
    and returns it as a structured DataFrame with positions and cumulative values.

    The DataFrame is particularly useful for:

    - Analyzing percentile distributions across different granularities
    - Visualizing step-function CDFs
    - Computing quantiles and inverse CDFs
    - Comparing empirical vs theoretical distributions

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

        - ``position``: Bin edge positions (sorted)
        - ``cumulative_probability``: CDF values at each position (0.0 to 1.0)
        - ``cumulative_frequency``: Cumulative frequency counts at each position
        - ``granularity``: Granularity level
        - ``is_best``: Boolean indicating the best histogram

        Each granularity level includes positions at both lower and upper bin edges
        to represent the step function.

    See Also
    --------
    ecdf : Get a callable ECDF object for evaluation at arbitrary points.
    ecdf_values : Get discrete CDF values as arrays.
    khisto.array.histogram_table : Histogram density DataFrame.

    Notes
    -----
    * The CDF is constructed by computing cumulative sums of bin values.
    * Position values include both lower bounds of bins and the final upper bound,
      creating a complete step-function representation.
    * For each granularity, the first position has cumulative_probability = 0.0
      and the last position has cumulative_probability = 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import ecdf_values_table
    >>> data = np.random.normal(0, 1, 1000)
    >>> cdf_df = ecdf_values_table(data, granularity="best")
    >>> # cdf_df has columns: position, cumulative_probability, cumulative_frequency, ...
    """
    # Import here to avoid circular dependency
    from khisto.array.histogram import histogram_table

    df = histogram_table(x, granularity=granularity)

    # Process each granularity level
    gran_df_list = [
        _create_granularity_cdf_df(df.filter(nw.col("granularity") == g))
        for g in df["granularity"].unique()
    ]

    return nw.concat(gran_df_list).sort("granularity")

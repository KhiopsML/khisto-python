"""Cumulative distribution functions for optimal histograms.

This module provides functions to compute cumulative distribution functions (CDF)
from optimal histogram bins. The CDF functions are separated from density-based
histogram functions to maintain clear API semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import narwhals as nw
import pyarrow as pa
from pyarrow import compute as pc

from khisto.core import compute_histogram

from ._shared import (
    build_edge_positions,
    extract_bin_edges,
    prepare_input,
    validate_granularity,
)

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoSeries


def _compute_cdf_values(df: pa.Table) -> pa.Array:
    """Compute cumulative distribution function values from histogram bins.

    Parameters
    ----------
    df : pa.Table
        Histogram table containing probability column.

    Returns
    -------
    pa.Array
        CDF values aligned with bin edges (prepended with 0.0).
    """
    probabilities = df["probability"].combine_chunks()
    cumsum = pc.cumulative_sum(probabilities)
    # cumsum[i] corresponds to probability up to and including bin i.
    # Build edge-aligned CDF: prepend 0.0, append final 1.0 already present as last cumsum.
    return pa.concat_arrays([pa.array([0.0]), cumsum])


def _compute_cdf_positions(df: pa.Table) -> pa.Array:
    """Compute position array for CDF values.

    Parameters
    ----------
    df : pa.Table
        Histogram table containing lower_bound and upper_bound columns.

    Returns
    -------
    pa.Array
        Complete position array including all bin edges.
    """
    lower_bounds, last_upper_bound = extract_bin_edges(df)
    return build_edge_positions(lower_bounds, last_upper_bound)


def cumulative_distribution(
    x: Union[ArrayT, IntoSeries],
    granularity: GranularityT = "best",
) -> tuple[ArrayT, ArrayT]:
    """Compute the cumulative distribution function (CDF) from optimal bins.

    The returned cumulative probabilities are aligned with the *bin edges* and
    thus have the same length. The first value is 0.0 and the last value is 1.0.

    This function computes the empirical CDF by accumulating probabilities from
    the optimal histogram bins. The CDF represents the probability that a random
    value from the distribution is less than or equal to each bin edge.

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
    cdf : ArrayT
        Cumulative probabilities at each bin edge (length = number_of_bins + 1).
        The first value is 0.0 and the last value is 1.0.
    positions: ArrayT
        Positions of the cdf values. length = number_of_bins + 1 = len(cdf).

    See Also
    --------
    cumulative_distribution_df : CDF as a DataFrame with additional metadata.
    khisto.array.histogram : Compute density histogram.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import cumulative_distribution
    >>> data = np.random.normal(0, 1, 1000)
    >>> cdf, edges = cumulative_distribution(data)
    >>> # cdf[0] ≈ 0.0, cdf[-1] = 1.0
    """
    arrow_array, backend = prepare_input(x)
    validate_granularity(granularity)

    df = compute_histogram(arrow_array, granularity=granularity)

    cdf = _compute_cdf_values(df)
    positions = _compute_cdf_positions(df)

    return backend.asarray(cdf), backend.asarray(positions)


def _create_granularity_cdf_df(gran_df: nw.DataFrame) -> nw.DataFrame:
    """Create CDF DataFrame for a single granularity level.

    Parameters
    ----------
    gran_df : nw.DataFrame
        Histogram DataFrame for one granularity level.

    Returns
    -------
    nw.DataFrame
        CDF DataFrame with position and cumulative_probability columns.
    """
    # Add cumulative probability column
    gran_df = gran_df.with_columns(
        nw.col("probability").cum_sum().alias("cumulative_probability")
    )

    # Drop unnecessary columns and prepare for edge alignment
    gran_df = gran_df.drop(["length", "frequency", "probability", "density", "center"])
    gran_df = gran_df.rename({"upper_bound": "position"})

    # Create first row with lower bound and 0.0 cumulative probability
    first_row = (
        gran_df.head(1)
        .with_columns(nw.lit(0.0).alias("cumulative_probability"))
        .drop("position")
        .rename({"lower_bound": "position"})
    )

    # Drop lower_bound from main dataframe and concatenate with first row
    gran_df = gran_df.drop("lower_bound")
    return nw.concat([first_row, gran_df])


def cumulative_distribution_df(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = None,
) -> nw.DataFrame:
    """Return cumulative distribution function (CDF) as a DataFrame.

    This function computes the empirical CDF from the optimal histogram bins
    and returns it as a structured DataFrame. Unlike
    :func:`~khisto.array.histogram.histogram_df`, this variant exposes bin
    positions (not bounds) and cumulative probabilities instead of densities.

    The CDF DataFrame is particularly useful for:

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
        - ``cumulative_probability``: CDF values at each position
        - ``granularity``: Granularity level
        - ``is_best``: Boolean indicating the best histogram

        Each granularity level includes positions at both lower and upper bin edges
        to represent the step function.

    See Also
    --------
    cumulative_distribution : CDF as arrays (cdf, bin_edges).
    khisto.array.histogram_df : Histogram density DataFrame.

    Notes
    -----
    * The CDF is constructed by computing cumulative sums of bin probabilities.
    * Position values include both lower bounds of bins and the final upper bound,
      creating a complete step-function representation.
    * For each granularity, the first position has cumulative_probability ≈ 0.0
      and the last position has cumulative_probability = 1.0.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import cumulative_distribution_df
    >>> data = np.random.normal(0, 1, 1000)
    >>> cdf_df = cumulative_distribution_df(data, granularity="best")
    >>> # cdf_df has columns: position, cumulative_probability, granularity, is_best
    """
    # Import here to avoid circular dependency
    from .histogram import histogram_df

    df = histogram_df(x, granularity=granularity)

    # Process each granularity level
    gran_df_list = [
        _create_granularity_cdf_df(df.filter(nw.col("granularity") == g))
        for g in df["granularity"].unique()
    ]

    return nw.concat(gran_df_list).sort("granularity")

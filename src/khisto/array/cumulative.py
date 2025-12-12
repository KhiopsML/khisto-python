"""Cumulative distribution functions for optimal histograms.

This module provides functions to compute cumulative distribution functions (CDF)
from optimal histogram bins. The CDF functions are separated from density-based
histogram functions to maintain clear API semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Union, cast, overload

import narwhals as nw
import numpy as np
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


# ============================================================================
# ECDF Class for evaluating CDF at arbitrary points
# ============================================================================


class ECDF:
    """Empirical Cumulative Distribution Function with linear interpolation.

    This class represents an ECDF computed from optimal histogram bins.
    It supports evaluation at arbitrary points using linear interpolation
    between the discrete CDF values at bin edges.

    Parameters
    ----------
    positions : np.ndarray
        Sorted array of bin edge positions.
    cdf_values : np.ndarray
        CDF values at each position (same length as positions).
    granularity : int
        The granularity level used to compute this ECDF.
    is_best : bool
        Whether this is the "best" granularity according to heuristics.

    Attributes
    ----------
    positions : np.ndarray
        The bin edge positions.
    cdf_values : np.ndarray
        The CDF values at each bin edge.
    granularity : int
        The granularity level.
    is_best : bool
        Whether this is the best granularity.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import ecdf
    >>> data = np.random.normal(0, 1, 1000)
    >>> cdf_func = ecdf(data)
    >>> # Evaluate at single point
    >>> cdf_func(0.0)
    >>> # Evaluate at multiple points
    >>> cdf_func(np.array([-1, 0, 1]))
    """

    def __init__(
        self,
        positions: np.ndarray,
        cdf_values: np.ndarray,
        granularity: int,
        is_best: bool,
    ) -> None:
        self._positions = positions
        self._cdf_values = cdf_values
        self._granularity = granularity
        self._is_best = is_best

    @property
    def positions(self) -> np.ndarray:
        """Bin edge positions."""
        return self._positions

    @property
    def cdf_values(self) -> np.ndarray:
        """CDF values at each bin edge."""
        return self._cdf_values

    @property
    def granularity(self) -> int:
        """Granularity level used for this ECDF."""
        return self._granularity

    @property
    def is_best(self) -> bool:
        """Whether this is the best granularity."""
        return self._is_best

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the ECDF at given point(s) using linear interpolation.

        Parameters
        ----------
        x : float or np.ndarray
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        float or np.ndarray
            CDF value(s) at the given point(s). Values outside the data range
            are clipped to 0.0 (below minimum) or 1.0 (above maximum).

        Examples
        --------
        >>> cdf_func = ecdf(data)
        >>> cdf_func(0.5)  # Single point
        0.723
        >>> cdf_func(np.array([0, 1, 2]))  # Multiple points
        array([0.5, 0.84, 0.98])
        """
        return self.evaluate(x)

    def evaluate(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate the ECDF at given point(s) using linear interpolation.

        Parameters
        ----------
        x : float or np.ndarray
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        float or np.ndarray
            CDF value(s) at the given point(s). Values outside the data range
            are clipped to 0.0 (below minimum) or 1.0 (above maximum).
        """
        return np.interp(x, self._positions, self._cdf_values)

    def __repr__(self) -> str:
        return (
            f"ECDF(granularity={self._granularity}, is_best={self._is_best}, "
            f"n_points={len(self._positions)}, "
            f"range=[{self._positions[0]:.4g}, {self._positions[-1]:.4g}])"
        )


class ECDFCollection:
    """Collection of ECDF objects for multiple granularity levels.

    This class holds multiple ECDF objects, one for each granularity level,
    and provides convenient access to them.

    Parameters
    ----------
    ecdfs : list[ECDF]
        List of ECDF objects, one per granularity level.

    Attributes
    ----------
    granularities : list[int]
        List of available granularity levels.
    best : ECDF
        The ECDF for the "best" granularity level.

    Examples
    --------
    >>> import numpy as np
    >>> from khisto.array import ecdf
    >>> data = np.random.normal(0, 1, 1000)
    >>> cdf_collection = ecdf(data, granularity=None)
    >>> # Access by granularity
    >>> cdf_collection[0]  # Granularity 0
    >>> cdf_collection[2]  # Granularity 2
    >>> # Access best
    >>> cdf_collection.best
    >>> # Evaluate best at a point
    >>> cdf_collection.best(0.0)
    """

    def __init__(self, ecdfs: list[ECDF]) -> None:
        self._ecdfs = {e.granularity: e for e in ecdfs}
        self._best = next((e for e in ecdfs if e.is_best), ecdfs[-1])
        self._granularities = sorted(self._ecdfs.keys())

    @property
    def granularities(self) -> list[int]:
        """List of available granularity levels."""
        return self._granularities

    @property
    def best(self) -> ECDF:
        """The ECDF for the best granularity level."""
        return self._best

    def __getitem__(self, granularity: int) -> ECDF:
        """Get ECDF for a specific granularity level.

        Parameters
        ----------
        granularity : int
            The granularity level.

        Returns
        -------
        ECDF
            The ECDF for that granularity.

        Raises
        ------
        KeyError
            If the granularity level doesn't exist.
        """
        if granularity not in self._ecdfs:
            raise KeyError(
                f"Granularity {granularity} not found. Available: {self._granularities}"
            )
        return self._ecdfs[granularity]

    def __iter__(self):
        """Iterate over ECDFs in order of granularity."""
        for g in self._granularities:
            yield self._ecdfs[g]

    def __len__(self) -> int:
        """Number of granularity levels."""
        return len(self._ecdfs)

    def __repr__(self) -> str:
        return (
            f"ECDFCollection(n_granularities={len(self._ecdfs)}, "
            f"granularities={self._granularities}, "
            f"best_granularity={self._best.granularity})"
        )


# ============================================================================
# Internal helper functions
# ============================================================================


def _compute_cdf_values(df: pa.Table, density: bool = True) -> pa.Array:
    """Compute cumulative distribution function values from histogram bins.

    Parameters
    ----------
    df : pa.Table
        Histogram table containing probability and frequency columns.
    density : bool, default True
        If True, accumulate probability values (result ranges from 0.0 to 1.0).
        If False, accumulate frequency counts.

    Returns
    -------
    pa.Array
        CDF values aligned with bin edges (prepended with 0.0 or 0).
    """
    column_name = "probability" if density else "frequency"
    values = df[column_name].combine_chunks()
    cumsum = pc.cumulative_sum(values)
    # cumsum[i] corresponds to cumulative value up to and including bin i.
    # Build edge-aligned CDF: prepend 0.0 (or 0 for frequency), append final value.
    if not density:
        return pa.concat_arrays([pa.array([0], type=values.type), cumsum])
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


# ============================================================================
# Public API: ecdf (returns callable ECDF object)
# ============================================================================


@overload
def ecdf(
    x: Union[ArrayT, IntoSeries],
    granularity: None,
) -> ECDFCollection: ...


@overload
def ecdf(
    x: Union[ArrayT, IntoSeries],
    granularity: Union[int, Literal["best"]] = ...,
) -> ECDF: ...


def ecdf(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = "best",
) -> Union[ECDF, ECDFCollection]:
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
    arrow_array, _ = prepare_input(x)

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
                ECDF(
                    positions=positions.to_numpy(),
                    cdf_values=cdf_vals.to_numpy(),
                    granularity=g,
                    is_best=(g == best_granularity),
                )
            )

        return ECDFCollection(ecdfs)

    # Single granularity
    cdf_vals = _compute_cdf_values(df, density=True)
    positions = _compute_cdf_positions(df)

    # Check if this is best
    is_best = df["is_best"][0].as_py() if "is_best" in df.column_names else True
    gran = df["granularity"][0].as_py() if "granularity" in df.column_names else 0

    return ECDF(
        positions=positions.to_numpy(),
        cdf_values=cdf_vals.to_numpy(),
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


def _create_granularity_cdf_df(gran_df: nw.DataFrame) -> nw.DataFrame:
    """Create CDF DataFrame for a single granularity level.

    Parameters
    ----------
    gran_df : nw.DataFrame
        Histogram DataFrame for one granularity level.

    Returns
    -------
    nw.DataFrame
        CDF DataFrame with position, cumulative_probability, and cumulative_frequency columns.
    """
    # Add cumulative probability and cumulative frequency columns
    gran_df = gran_df.with_columns(
        nw.col("probability").cum_sum().alias("cumulative_probability"),
        nw.col("frequency").cum_sum().alias("cumulative_frequency"),
    )

    # Drop unnecessary columns and prepare for edge alignment
    gran_df = gran_df.drop(["length", "frequency", "probability", "density", "center"])
    gran_df = gran_df.rename({"upper_bound": "position"})

    # Create first row with lower bound and 0 cumulative values
    first_row = (
        gran_df.head(1)
        .with_columns(
            nw.lit(0.0).alias("cumulative_probability"),
            nw.lit(0).cast(nw.Int64).alias("cumulative_frequency"),
        )
        .drop("position")
        .rename({"lower_bound": "position"})
    )

    # Drop lower_bound from main dataframe and concatenate with first row
    gran_df = gran_df.drop("lower_bound")
    return nw.concat([first_row, gran_df])


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
    from .histogram import histogram_table

    df = histogram_table(x, granularity=granularity)

    # Process each granularity level
    gran_df_list = [
        _create_granularity_cdf_df(df.filter(nw.col("granularity") == g))
        for g in df["granularity"].unique()
    ]

    return nw.concat(gran_df_list).sort("granularity")

"""Core classes for cumulative distribution functions."""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Union

from .._shared import prepare_input

if TYPE_CHECKING:
    from khisto.typing import ArrayT


# ============================================================================
# ECDF Class for evaluating CDF at arbitrary points
# ============================================================================


class ECDFResult:
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
        positions: ArrayT,
        cdf_values: ArrayT,
        granularity: int,
        is_best: bool,
    ) -> None:
        self._positions = positions
        self._cdf_values = cdf_values
        self._granularity = granularity
        self._is_best = is_best

    @property
    def positions(self) -> ArrayT:
        """Bin edge positions."""
        return self._positions

    @property
    def cdf_values(self) -> ArrayT:
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

    def __call__(self, x: Union[float, ArrayT]) -> Union[float, ArrayT]:
        """Evaluate the ECDF at given point(s) using linear interpolation.

        Parameters
        ----------
        x : float or ArrayT
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        float or ArrayT
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

    def evaluate(self, x: Union[float, ArrayT]) -> Union[float, ArrayT]:
        """Evaluate the ECDF at given point(s) using linear interpolation.

        Parameters
        ----------
        x : float or ArrayT
            Point(s) at which to evaluate the CDF.

        Returns
        -------
        float or ArrayT
            CDF value(s) at the given point(s). Values outside the data range
            are clipped to 0.0 (below minimum) or 1.0 (above maximum).
        """
        arrow_array, backend = prepare_input(x)

        result_list = []

        for val in arrow_array.to_pylist():
            if val is None:
                result_list.append(None)
                continue

            if val <= self._positions[0]:
                result_list.append(0.0)
                continue
            if val >= self._positions[-1]:
                result_list.append(1.0)
                continue

            # Find index i such that pos[i] <= val < pos[i+1]
            idx = bisect.bisect_right(self._positions, val)
            # idx is the insertion point to maintain order.
            # pos[idx-1] <= val < pos[idx]

            p0 = self._positions[idx - 1]
            p1 = self._positions[idx]
            v0 = self._cdf_values[idx - 1]
            v1 = self._cdf_values[idx]

            # Linear interpolation
            t = (val - p0) / (p1 - p0)
            res = v0 + t * (v1 - v0)
            result_list.append(res)

        return backend.asarray(result_list)

    def __repr__(self) -> str:
        return (
            f"ECDF(granularity={self._granularity}, is_best={self._is_best}, "
            f"n_points={len(self._positions)}, "
            f"range=[{self._positions[0]:.4g}, {self._positions[-1]:.4g}])"
        )


class ECDFResultCollection:
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

    def __init__(self, ecdfs: list[ECDFResult]) -> None:
        self._ecdfs = {e.granularity: e for e in ecdfs}
        self._best = next((e for e in ecdfs if e.is_best), ecdfs[-1])
        self._granularities = sorted(self._ecdfs.keys())

    @property
    def granularities(self) -> list[int]:
        """List of available granularity levels."""
        return self._granularities

    @property
    def best(self) -> ECDFResult:
        """The ECDF for the best granularity level."""
        return self._best

    def __getitem__(self, granularity: int) -> ECDFResult:
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

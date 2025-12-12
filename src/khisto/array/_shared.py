"""Shared utilities for array histogram and cumulative distribution functions.

This internal module provides common functionality used by both histogram
and cumulative distribution functions to reduce code duplication.
"""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, Union

import narwhals as nw
import pyarrow as pa

from khisto.utils import get_array_backend, parse_narwhals_series, to_arrow

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoSeries


def prepare_input(x: Union[ArrayT, IntoSeries]) -> tuple[pa.Array, ModuleType]:
    """Parse and convert input data to Arrow format while preserving backend info.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data to parse.

    Returns
    -------
    arrow_array : pa.Array
        Data converted to PyArrow array.
    backend : object
        The original array backend for result conversion.
    """
    backend = get_array_backend(x)

    if isinstance(x, (float, int, complex)):
        x = [x]
    arrow_array = to_arrow(x)
    return arrow_array, backend


def validate_granularity(granularity: Optional[GranularityT]) -> None:
    """Validate that granularity parameter.

    Parameters
    ----------
    granularity : int, "best", or None
        The granularity value to validate.

    Raises
    ------
    ValueError
        If granularity is None.
    """
    if isinstance(granularity, (float, int)):
        if granularity < 0:
            raise ValueError("Granularity cannot be negative.")
    elif isinstance(granularity, str):
        if granularity != "best":
            raise ValueError('Granularity string must be "best".')
    elif granularity is not None:
        raise ValueError("Granularity must be an integer, 'best', or None.")


def extract_bin_edges(df: pa.Table) -> tuple[pa.Array, float]:
    """Extract bin edges from histogram dataframe.

    Parameters
    ----------
    df : pa.Table
        Histogram table from compute_histogram.

    Returns
    -------
    lower_bounds : pa.Array
        Combined lower bound values.
    last_upper_bound : float
        The final upper bound value.
    """
    lower_bounds = df["lower_bound"].combine_chunks()
    last_upper_bound = df["upper_bound"][-1].as_py()
    return lower_bounds, last_upper_bound


def build_edge_positions(lower_bounds: pa.Array, last_upper_bound: float) -> pa.Array:
    """Build complete position array from bin bounds.

    Parameters
    ----------
    lower_bounds : pa.Array
        Array of lower bin bounds.
    last_upper_bound : float
        The final upper bound.

    Returns
    -------
    pa.Array
        Concatenated array of all bin edge positions.
    """
    return pa.concat_arrays([lower_bounds, pa.array([last_upper_bound])])


def prepare_input_for_df(x: Union[ArrayT, IntoSeries]) -> tuple[pa.Array, Any]:
    """Parse input for DataFrame functions, preserving Narwhals backend.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Input data to parse.

    Returns
    -------
    arrow_array : pa.Array
        Data converted to PyArrow array.
    narwhals_backend : object
        The Narwhals backend (pa for non-series, native namespace for series).
    """
    backend = pa
    series = parse_narwhals_series(x)
    if series is not None:
        backend = nw.get_native_namespace(series)
        x = series.to_arrow()
    arrow_array = to_arrow(x)
    return arrow_array, backend

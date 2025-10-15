from __future__ import annotations

from typing import TYPE_CHECKING, Union


from khisto.core import compute_histogram
import pyarrow as pa
import narwhals as nw
from khisto.utils import get_array_backend, parse_narwhals_series, to_arrow

if TYPE_CHECKING:
    from khisto.typing import ArrayT
    from narwhals.typing import IntoSeries


def histogram(x: Union[ArrayT, IntoSeries]) -> tuple[ArrayT, ArrayT]:
    """Compute histogram of an array.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Array supporting the Python Array API standard, or a list, tuple, array.array,
        or Narwhals Series.

    Returns
    -------
    densities : ArrayT
        Array of density values for each bin.
    bin_edges : ArrayT
        Sorted array of bin edges (lower and upper bounds).
    """
    # Detect narwhals Series by type or attribute
    series = parse_narwhals_series(x)
    if series is not None:
        x = series.to_arrow()
    backend = get_array_backend(x)
    x = to_arrow(x)
    df = compute_histogram(x, only_best=True)

    lower_bounds = df["lower_bound"]
    last_upper_bound = df["upper_bound"][-1]
    densities = df["density"]

    bin_edges = pa.concat_arrays(
        [lower_bounds.combine_chunks(), pa.array([last_upper_bound])]
    ).sort()

    bin_edges = backend.asarray(bin_edges)
    densities = backend.asarray(densities.combine_chunks())
    return densities, bin_edges


def histogram_bin_edges(x: Union[ArrayT, IntoSeries]) -> ArrayT:
    """Compute histogram bin edges of an array.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Array supporting the Python Array API standard, or a list, tuple, array.array,
        or Narwhals Series.

    Returns
    -------
    ArrayT
        Sorted array of bin edges (lower and upper bounds).
    """
    series = parse_narwhals_series(x)
    if series is not None:
        x = series.to_list()
    backend = get_array_backend(x)
    x = to_arrow(x)
    df = compute_histogram(x, only_best=True)

    lower_bounds = df["lower_bound"]
    last_upper_bound = df["upper_bound"][-1]

    bin_edges = pa.concat_arrays(
        [lower_bounds.combine_chunks(), pa.array([last_upper_bound])]
    ).sort()

    bin_edges = backend.asarray(bin_edges)
    return bin_edges


def histogram_series(
    x: Union[ArrayT, IntoSeries], only_best: bool = False
) -> nw.DataFrame:
    """Compute histogram series of an array.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Array supporting the Python Array API standard, or a list, tuple, array.array,
        or Narwhals Series.
    only_best : bool, default False
        If True, return only the best histogram.
        If False, return all histograms with granularity information.

    Returns
    -------
    nw.DataFrame
        A Narwhals DataFrame with histogram data including columns:
        - lower_bound : float64
            Lower bound of each bin.
        - upper_bound : float64
            Upper bound of each bin.
        - length : float64
            Length of each bin.
        - frequency : int64
            Frequency count in each bin.
        - probability : float64
            Probability of each bin.
        - density : float64
            Density of each bin.
        - granularity : int32
            Histogram granularity level (if only_best=False).
        - is_best : bool
            Whether this is the best histogram (if only_best=False).
    """
    backend = pa
    series = parse_narwhals_series(x)
    if series is not None:
        backend = nw.get_native_namespace(series)
        x = series.to_arrow()
    x = to_arrow(x)
    df = compute_histogram(x, only_best=only_best)

    df = nw.from_arrow(df, backend=backend)
    return df

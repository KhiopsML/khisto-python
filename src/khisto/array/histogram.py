from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union


from khisto.core import compute_histogram
import pyarrow as pa
from pyarrow import compute as pc
import narwhals as nw
from khisto.utils import get_array_backend, parse_narwhals_series, to_arrow

if TYPE_CHECKING:
    from khisto.typing import ArrayT, GranularityT
    from narwhals.typing import IntoSeries


def histogram(
    x: Union[ArrayT, IntoSeries],
    cumulative: bool = False,
    granularity: GranularityT = "best",
) -> tuple[ArrayT, ArrayT]:
    """Compute histogram of an array.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Array supporting the Python Array API standard, or a list, tuple, array.array,
        or Narwhals Series.
    cumulative : bool, default False
        If True, return cumulative probability distribution instead of densities.
        The cumulative values are computed by taking into account the length of each bin.
        The returned densities start with 0.0 as the first value and end with 1.0 with
        as many values as there are bin edges.
    granularity : int or "best", default "best"
        Granularity level to use. If "best", uses the best histogram.
        If an integer, uses the histogram at that granularity level.
        If the provided granularity is higher than the most granular, uses the most granular.

    Returns
    -------
    densities : ArrayT
        Array of density values for each bin (or cumulative probabilities if cumulative=True).
    bin_edges : ArrayT
        Sorted array of bin edges (lower and upper bounds).
    """
    # Detect narwhals Series by type or attribute
    series = parse_narwhals_series(x)
    if series is not None:
        x = series.to_arrow()
    backend = get_array_backend(x)
    x = to_arrow(x)

    if granularity is None:
        raise ValueError("granularity must be specified as an integer or 'best'")
    df = compute_histogram(x, granularity=granularity)

    lower_bounds = df["lower_bound"]
    last_upper_bound = df["upper_bound"][-1]

    if cumulative:
        densities = pc.cumulative_sum(df["probability"])
        densities = pa.concat_arrays([pa.array([0.0]), densities.combine_chunks()])
    else:
        densities = df["density"].combine_chunks()

    bin_edges = pa.concat_arrays(
        [lower_bounds.combine_chunks(), pa.array([last_upper_bound])]
    ).sort()

    bin_edges = backend.asarray(bin_edges)
    densities = backend.asarray(densities)
    return densities, bin_edges


def histogram_bin_edges(
    x: Union[ArrayT, IntoSeries], granularity: GranularityT = "best"
) -> ArrayT:
    """Compute histogram bin edges of an array.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Array supporting the Python Array API standard, or a list, tuple, array.array,
        or Narwhals Series.
    granularity : int or "best", default "best"
        Granularity level to use. If "best", uses the best histogram.
        If an integer, uses the histogram at that granularity level.
        If the provided granularity is higher than the most granular, uses the most granular.

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

    if granularity is None:
        raise ValueError("granularity must be specified as an integer or 'best'")
    df = compute_histogram(x, granularity=granularity)

    lower_bounds = df["lower_bound"]
    last_upper_bound = df["upper_bound"][-1]

    bin_edges = pa.concat_arrays(
        [lower_bounds.combine_chunks(), pa.array([last_upper_bound])]
    ).sort()

    bin_edges = backend.asarray(bin_edges)
    return bin_edges


def histogram_series(
    x: Union[ArrayT, IntoSeries],
    granularity: Optional[GranularityT] = None,
    cumulative: bool = False,
) -> nw.DataFrame:
    """Compute histogram series of an array.

    Parameters
    ----------
    x : ArrayT or IntoSeries
        Array supporting the Python Array API standard, or a list, tuple, array.array,
        or Narwhals Series.
    granularity : int or 'best', optional
        Desired histogram granularity level.
        If None, all granularities are computed.
        If 'best', only the best histogram is computed.
        If an integer, the histogram with that granularity is returned.
        If the provided granularity is higher than the most granular, uses the most granular.
    cumulative : bool, default False
        If True, add a cumulative_probability column to the output.
        The cumulative values are computed by taking into account the length of each bin.

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
        - cumulative_probability : float64
            Cumulative probability (if cumulative=True).
        - granularity : int32
            Histogram granularity level. Start at 0 for the least granular.
        - is_best : bool
            Whether this is the best histogram.
    """
    backend = pa
    series = parse_narwhals_series(x)
    if series is not None:
        backend = nw.get_native_namespace(series)
        x = series.to_arrow()
    x = to_arrow(x)
    df = compute_histogram(x, granularity=granularity)
    df = nw.from_arrow(df, backend=backend)

    if cumulative:
        # Compute cumulative sum within each granularity group
        # Since data is sorted by granularity, we can use a manual approach
        cumsum_parts = []
        for granularity in df["granularity"].unique(maintain_order=True):
            group_df = df.filter(nw.col("granularity") == granularity)
            group_df = group_df.with_columns(
                nw.col("probability").cum_sum().alias("cumulative_probability")
            )
            cumsum_parts.append(group_df)
        df = nw.concat(cumsum_parts)
    return df

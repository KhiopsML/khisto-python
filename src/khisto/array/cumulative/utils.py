"""Internal helper functions for cumulative distribution functions."""

from __future__ import annotations

import narwhals as nw
import pyarrow as pa
import pyarrow.compute as pc

from .._shared import build_edge_positions, extract_bin_edges


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

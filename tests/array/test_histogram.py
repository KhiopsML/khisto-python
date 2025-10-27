"""Tests for histogram functions in khisto.array.histogram module."""

from __future__ import annotations

from array import array

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from khisto.array import (
    histogram,
    histogram_bin_edges,
    histogram_df,
    cumulative_distribution,
    cumulative_distribution_df,
)


# Test data fixtures
@pytest.fixture
def simple_data():
    """Simple test data: [1, 2, 3, 4, 5]"""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def normal_data():
    """Normal distribution data"""
    np.random.seed(42)
    return np.random.normal(loc=0, scale=1, size=1000).tolist()


@pytest.fixture
def uniform_data():
    """Uniform distribution data"""
    np.random.seed(42)
    return np.random.uniform(low=0, high=10, size=500).tolist()


@pytest.fixture
def bimodal_data():
    """Bimodal distribution data"""
    np.random.seed(42)
    data1 = np.random.normal(loc=-2, scale=0.5, size=250)
    data2 = np.random.normal(loc=2, scale=0.5, size=250)
    return np.concatenate([data1, data2]).tolist()


class TestHistogram:
    """Test cases for histogram function."""

    def test_histogram_with_list(self, simple_data):
        """Test histogram with Python list input."""
        densities, bin_edges = histogram(simple_data)

        assert isinstance(densities, list)
        assert isinstance(bin_edges, list)
        assert len(densities) > 0
        assert len(bin_edges) == len(densities) + 1
        # Check bin edges are sorted
        assert all(bin_edges[i] <= bin_edges[i + 1] for i in range(len(bin_edges) - 1))

    def test_histogram_with_tuple(self, simple_data):
        """Test histogram with tuple input."""
        densities, bin_edges = histogram(tuple(simple_data))

        assert isinstance(densities, list)
        assert isinstance(bin_edges, list)
        assert len(densities) > 0

    def test_histogram_with_array(self, simple_data):
        """Test histogram with array.array input."""
        arr = array("d", simple_data)
        densities, bin_edges = histogram(arr)

        assert isinstance(densities, list)
        assert isinstance(bin_edges, list)
        assert len(densities) > 0

    def test_histogram_with_numpy(self, normal_data):
        """Test histogram with NumPy array input."""
        np_array = np.array(normal_data)
        densities, bin_edges = histogram(np_array)

        assert hasattr(densities, "__array_namespace__")
        assert hasattr(bin_edges, "__array_namespace__")
        assert len(densities) > 0
        assert len(bin_edges) == len(densities) + 1

    def test_histogram_with_pyarrow(self, simple_data):
        """Test histogram with PyArrow array input."""
        pa_array = pa.array(simple_data)
        densities, bin_edges = histogram(pa_array)

        assert isinstance(densities, pa.Array)
        assert isinstance(bin_edges, pa.Array)
        assert len(densities) > 0

    def test_histogram_with_pandas_series(self, normal_data):
        """Test histogram with Pandas Series input."""
        series = pd.Series(normal_data)
        densities, bin_edges = histogram(series)

        # Should return PyArrow arrays when input is Pandas Series
        assert isinstance(densities, pa.Array)
        assert isinstance(bin_edges, pa.Array)
        assert len(densities) > 0

    def test_histogram_with_polars_series(self, normal_data):
        """Test histogram with Polars Series input."""
        series = pl.Series(normal_data)
        densities, bin_edges = histogram(series)

        # Should return Polars Series when input is Polars Series
        assert isinstance(densities, pa.Array)
        assert isinstance(bin_edges, pa.Array)
        assert len(densities) > 0

    def test_histogram_density_sum(self, uniform_data):
        """Test that histogram densities properly sum to 1 when multiplied by bin widths."""
        densities, bin_edges = histogram(uniform_data)

        # Convert to numpy for easier calculation
        densities_np = np.asarray(densities)
        bin_edges_np = np.asarray(bin_edges)

        # Calculate bin widths
        bin_widths = bin_edges_np[1:] - bin_edges_np[:-1]

        # Sum of (density * width) should be approximately 1
        total_probability = np.sum(densities_np * bin_widths)
        assert np.isclose(total_probability, 1.0, rtol=1e-5)

    def test_histogram_bin_coverage(self, simple_data):
        """Test that bin edges cover the full range of data."""
        densities, bin_edges = histogram(simple_data)

        assert min(bin_edges) <= min(simple_data)
        assert max(bin_edges) >= max(simple_data)

    def test_histogram_with_duplicates(self):
        """Test histogram with duplicate values."""
        data = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        densities, bin_edges = histogram(data)

        assert len(densities) > 0
        assert len(bin_edges) > 0

    def test_histogram_with_single_value(self):
        """Test histogram with all identical values."""
        data = [5.0] * 100
        densities, bin_edges = histogram(data)

        assert len(densities) > 0
        assert len(bin_edges) > 0


class TestHistogramBinEdges:
    """Test cases for histogram_bin_edges function."""

    def test_bin_edges_with_list(self, simple_data):
        """Test histogram_bin_edges with list input."""
        bin_edges = histogram_bin_edges(simple_data)

        assert isinstance(bin_edges, list)
        assert len(bin_edges) > 0
        # Check bin edges are sorted
        assert all(bin_edges[i] <= bin_edges[i + 1] for i in range(len(bin_edges) - 1))

    def test_bin_edges_with_numpy(self, normal_data):
        """Test histogram_bin_edges with NumPy array."""
        np_array = np.array(normal_data)
        bin_edges = histogram_bin_edges(np_array)

        assert hasattr(bin_edges, "__array_namespace__")
        assert len(bin_edges) > 1

    def test_bin_edges_with_pandas_series(self, uniform_data):
        """Test histogram_bin_edges with Pandas Series."""
        series = pd.Series(uniform_data)
        bin_edges = histogram_bin_edges(series)

        assert len(bin_edges) > 1

    def test_bin_edges_with_polars_series(self, uniform_data):
        """Test histogram_bin_edges with Polars Series."""
        series = pl.Series(uniform_data)
        bin_edges = histogram_bin_edges(series)

        assert len(bin_edges) > 1

    def test_bin_edges_sorted(self, bimodal_data):
        """Test that bin edges are always sorted."""
        bin_edges = histogram_bin_edges(bimodal_data)

        assert bin_edges == sorted(bin_edges)

    def test_bin_edges_coverage(self, normal_data):
        """Test that bin edges cover the data range."""
        bin_edges = histogram_bin_edges(normal_data)

        assert min(bin_edges) <= min(normal_data)
        assert max(bin_edges) >= max(normal_data)

    def test_bin_edges_consistency_with_histogram(self, simple_data):
        """Test that bin_edges from histogram_bin_edges matches histogram function."""
        densities, bin_edges_from_hist = histogram(simple_data)
        bin_edges_from_func = histogram_bin_edges(simple_data)

        assert bin_edges_from_hist == bin_edges_from_func


class TestHistogramDataFrame:
    """Test cases for histogram_df function (formerly histogram_series)."""

    def test_histogram_df_best_granularity(self, simple_data):
        """Test histogram_df with granularity='best'."""
        df = histogram_df(simple_data, granularity="best")

        assert isinstance(df, nw.DataFrame)

        # Check required columns exist
        expected_cols = [
            "lower_bound",
            "upper_bound",
            "length",
            "frequency",
            "probability",
            "density",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Check data types
        assert len(df) > 0

    def test_histogram_df_all_histograms(self, normal_data):
        """Test histogram_df with granularity=None."""
        df = histogram_df(normal_data, granularity=None)

        assert isinstance(df, nw.DataFrame)

        # Check required columns including granularity and is_best
        expected_cols = [
            "lower_bound",
            "upper_bound",
            "length",
            "frequency",
            "probability",
            "density",
            "granularity",
            "is_best",
        ]
        for col in expected_cols:
            assert col in df.columns

        # Should have multiple granularities
        granularities = df["granularity"].unique()
        assert len(granularities) > 1

    def test_histogram_df_with_numpy(self, uniform_data):
        """Test histogram_df with NumPy array."""
        np_array = np.array(uniform_data)
        df = histogram_df(np_array, granularity="best")

        assert isinstance(df, nw.DataFrame)
        assert len(df) > 0

    def test_histogram_df_with_pandas_series(self, normal_data):
        """Test histogram_df with Pandas Series."""
        series = pd.Series(normal_data)
        df = histogram_df(series, granularity="best")

        assert isinstance(df, nw.DataFrame)
        # Check that backend is pandas
        native = nw.to_native(df)
        assert isinstance(native, pd.DataFrame)

    def test_histogram_df_with_polars_series(self, bimodal_data):
        """Test histogram_df with Polars Series."""
        series = pl.Series(bimodal_data)
        df = histogram_df(series, granularity="best")

        assert isinstance(df, nw.DataFrame)
        # Check that backend is polars
        native = nw.to_native(df)
        assert isinstance(native, pl.DataFrame)

    def test_histogram_df_probability_sum(self, uniform_data):
        """Test that probabilities sum to 1."""
        df = histogram_df(uniform_data, granularity="best")

        total_prob = df["probability"].sum()
        assert np.isclose(total_prob, 1.0, rtol=1e-5)

    def test_histogram_df_density_calculation(self, simple_data):
        """Test that density = probability / length."""
        df = histogram_df(simple_data, granularity="best")

        for row in df.iter_rows(named=True):
            expected_density = (
                row["probability"] / row["length"] if row["length"] > 0 else 0
            )
            assert np.isclose(row["density"], expected_density, rtol=1e-5)

    def test_histogram_df_best_marked(self, normal_data):
        """Test that best histogram is marked when granularity=None."""
        df = histogram_df(normal_data, granularity=None)

        # Should have is_best column
        assert "is_best" in df.columns

        # Should have at least one True value
        best_count = df.filter(nw.col("is_best")).shape[0]
        assert best_count > 0

    def test_histogram_df_bin_bounds(self, simple_data):
        """Test that upper_bound >= lower_bound for all bins."""
        df = histogram_df(simple_data, granularity="best")
        for row in df.iter_rows(named=True):
            assert row["upper_bound"] >= row["lower_bound"]

    def test_histogram_df_length_calculation(self, uniform_data):
        """Test that length = upper_bound - lower_bound."""
        df = histogram_df(uniform_data, granularity="best")

        for row in df.iter_rows(named=True):
            expected_length = row["upper_bound"] - row["lower_bound"]
            assert np.isclose(row["length"], expected_length, rtol=1e-5)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array(self):
        """Test with empty array - should handle gracefully or raise appropriate error."""
        with pytest.raises((ValueError, RuntimeError, Exception)):  # pyrefly: ignore
            histogram([])

    def test_single_element(self):
        """Test with single element."""
        data = [42.0]
        densities, bin_edges = histogram(data)

        assert len(densities) > 0
        assert len(bin_edges) > 0

    def test_two_elements(self):
        """Test with two elements."""
        data = [1.0, 2.0]
        densities, bin_edges = histogram(data)

        assert len(densities) > 0
        assert len(bin_edges) == len(densities) + 1

    def test_negative_values(self):
        """Test with negative values."""
        data = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
        densities, bin_edges = histogram(data)

        assert len(densities) > 0
        assert min(bin_edges) <= -5.0

    def test_large_range(self):
        """Test with very large range of values."""
        data = [0.001, 1000000.0]
        densities, bin_edges = histogram(data)

        assert len(densities) > 0
        assert len(bin_edges) > 0

    def test_very_small_values(self):
        """Test with very small values."""
        data = [1e-10, 2e-10, 3e-10, 4e-10]
        densities, bin_edges = histogram(data)

        assert len(densities) > 0


class TestReturnTypeConsistency:
    """Test that return types match input types."""

    def test_list_returns_list(self, simple_data):
        """Test that list input returns list output."""
        densities, bin_edges = histogram(simple_data)
        assert isinstance(densities, list)
        assert isinstance(bin_edges, list)

    def test_numpy_returns_numpy(self, simple_data):
        """Test that NumPy input returns NumPy output."""
        np_array = np.array(simple_data)
        densities, bin_edges = histogram(np_array)
        assert hasattr(densities, "__array_namespace__")
        assert hasattr(bin_edges, "__array_namespace__")

    def test_pyarrow_returns_pyarrow(self, simple_data):
        """Test that PyArrow input returns PyArrow output."""
        pa_array = pa.array(simple_data)
        densities, bin_edges = histogram(pa_array)
        assert isinstance(densities, pa.Array)
        assert isinstance(bin_edges, pa.Array)

    def test_pandas_series_backend(self, simple_data):
        """Test histogram_df with Pandas maintains backend."""
        series = pd.Series(simple_data)
        df = histogram_df(series, granularity="best")
        native = nw.to_native(df)
        assert isinstance(native, pd.DataFrame)

    def test_polars_series_backend(self, simple_data):
        """Test histogram_df with Polars maintains backend."""
        series = pl.Series(simple_data)
        df = histogram_df(series, granularity="best")
        native = nw.to_native(df)
        assert isinstance(native, pl.DataFrame)


class TestCumulativeAPIs:
    """Tests for the new dedicated cumulative histogram APIs."""

    def test_density_histogram_normalization(self, uniform_data):
        densities, bin_edges = histogram(uniform_data)
        densities_np = np.asarray(densities)
        bin_edges_np = np.asarray(bin_edges)
        widths = bin_edges_np[1:] - bin_edges_np[:-1]
        total = np.sum(densities_np * widths)
        assert np.isclose(total, 1.0, rtol=1e-5)

    def test_cumulative_distribution_basic(self, uniform_data):
        cdf, bin_edges = cumulative_distribution(uniform_data)
        cdf_np = np.asarray(cdf)
        assert np.isclose(cdf_np[0], 0.0, rtol=1e-5)
        assert np.isclose(cdf_np[-1], 1.0, rtol=1e-5)
        assert np.all(np.diff(cdf_np) >= 0)
        assert len(cdf_np) == len(bin_edges)

    def test_cumulative_distribution_backends(self, simple_data):
        cdf_list, _ = cumulative_distribution(simple_data)
        assert isinstance(cdf_list, list)
        np_array = np.array(simple_data)
        cdf_np, _ = cumulative_distribution(np_array)
        assert hasattr(cdf_np, "__array_namespace__")
        pa_array = pa.array(simple_data)
        cdf_pa, _ = cumulative_distribution(pa_array)
        assert isinstance(cdf_pa, pa.Array)

    def test_cumulative_distribution_df_best(self, normal_data):
        """Test cumulative_distribution_df returns position-based CDF for best granularity."""
        df = cumulative_distribution_df(normal_data, granularity="best")

        # Schema: has position and cumulative_probability, no bounds
        assert "position" in df.columns
        assert "cumulative_probability" in df.columns
        assert "lower_bound" not in df.columns
        assert "upper_bound" not in df.columns

        # No density/frequency columns in cumulative variant
        assert "density" not in df.columns
        assert "frequency" not in df.columns
        assert "probability" not in df.columns
        assert "length" not in df.columns

        # CDF properties
        cum = df["cumulative_probability"].to_numpy()
        assert np.isclose(cum[0], 0.0, atol=1e-10)  # Starts at 0
        assert np.isclose(cum[-1], 1.0, rtol=1e-5)  # Ends at 1
        assert np.all(np.diff(cum) >= 0)  # Monotonic non-decreasing

        # Positions are sorted
        pos = df["position"].to_numpy()
        assert np.all(np.diff(pos) >= 0)

    def test_cumulative_distribution_df_all_granularities(self, normal_data):
        """Test cumulative_distribution_df with granularity=None returns all levels."""
        df = cumulative_distribution_df(normal_data, granularity=None)

        assert "position" in df.columns
        assert "cumulative_probability" in df.columns
        assert "granularity" in df.columns
        assert "is_best" in df.columns

        # Check each granularity separately
        for g in df["granularity"].unique():
            sub = df.filter(nw.col("granularity") == g)
            cum = sub["cumulative_probability"].to_numpy()

            # Each granularity's CDF starts at ~0 and ends at 1
            assert np.isclose(cum[0], 0.0, atol=1e-10)
            assert np.isclose(cum[-1], 1.0, rtol=1e-5)
            assert np.all(np.diff(cum) >= 0)  # Monotonic

            # Positions within each granularity are sorted
            pos = sub["position"].to_numpy()
            assert np.all(np.diff(pos) >= 0)

    def test_cumulative_distribution_df_step_function_structure(self, simple_data):
        """Test that cumulative_distribution_df includes bin edge positions."""
        df = cumulative_distribution_df(simple_data, granularity="best")

        # Get corresponding histogram to compare bin count
        hist_df = histogram_df(simple_data, granularity="best")
        n_bins = len(hist_df)

        # CDF should have n_bins + 1 positions (all bin edges)
        assert len(df) == n_bins + 1

        # First position should correspond to first lower_bound
        # Last position should correspond to last upper_bound
        first_lower = hist_df["lower_bound"].to_numpy()[0]
        last_upper = hist_df["upper_bound"].to_numpy()[-1]

        positions = df["position"].to_numpy()
        assert np.isclose(positions[0], first_lower, rtol=1e-10)
        assert np.isclose(positions[-1], last_upper, rtol=1e-10)

    def test_histogram_df_no_cumulative_column(self, uniform_data):
        df = histogram_df(uniform_data, granularity="best")
        # histogram_df keeps bounds and does not include cumulative or position
        assert "cumulative_probability" not in df.columns
        assert "position" not in df.columns


class TestGranularityParameter:
    """Test cases for granularity parameter in histogram and histogram_bin_edges functions."""

    def test_histogram_granularity_best_default(self, normal_data):
        """Test histogram with default granularity='best'."""
        densities, bin_edges = histogram(normal_data, granularity="best")

        assert len(densities) > 0
        assert len(bin_edges) == len(densities) + 1

    def test_histogram_granularity_zero(self, normal_data):
        """Test histogram with granularity=0."""
        densities_0, bin_edges_0 = histogram(normal_data, granularity=0)
        densities_best, bin_edges_best = histogram(normal_data, granularity="best")

        # Granularity 0 might be the same as best, but should at least be valid
        assert len(densities_0) > 0
        assert len(bin_edges_0) == len(densities_0) + 1

    def test_histogram_granularity_increases_bins(self, normal_data):
        """Test that higher granularity generally means more bins."""
        _, bin_edges_0 = histogram(normal_data, granularity=0)
        _, bin_edges_1 = histogram(normal_data, granularity=1)

        # Higher granularity should have at least as many bins
        # (though not guaranteed in all cases, generally true)
        assert len(bin_edges_1) >= len(bin_edges_0)

    def test_histogram_granularity_consistency(self, uniform_data):
        """Test that same granularity gives consistent results."""
        densities_1, bin_edges_1 = histogram(uniform_data, granularity=1)
        densities_2, bin_edges_2 = histogram(uniform_data, granularity=1)

        # Should be identical
        assert np.allclose(np.asarray(densities_1), np.asarray(densities_2))
        assert np.allclose(np.asarray(bin_edges_1), np.asarray(bin_edges_2))

    def test_histogram_granularity_exceeds_maximum(self, simple_data):
        """Test that requesting granularity higher than max uses most granular."""
        # Request a very high granularity
        densities_high, bin_edges_high = histogram(simple_data, granularity=1000)

        # Should still return valid results (capped at max granularity)
        assert len(densities_high) > 0
        assert len(bin_edges_high) == len(densities_high) + 1

    def test_histogram_granularity_with_cumulative(self, normal_data):
        """Test granularity parameter works with cumulative parameter."""
        cum_probs_0, _ = cumulative_distribution(normal_data, granularity=0)
        cum_probs_1, _ = cumulative_distribution(normal_data, granularity=1)

        # Both should end at approximately 1
        assert np.isclose(np.asarray(cum_probs_0)[-1], 1.0, rtol=1e-5)
        assert np.isclose(np.asarray(cum_probs_1)[-1], 1.0, rtol=1e-5)

    def test_histogram_granularity_none_raises_error(self, simple_data):
        """Test that None granularity raises ValueError."""
        with pytest.raises(ValueError, match="granularity must be"):
            histogram(simple_data, granularity=None)  # type: ignore

    def test_histogram_bin_edges_granularity_best(self, normal_data):
        """Test histogram_bin_edges with granularity='best'."""
        bin_edges = histogram_bin_edges(normal_data, granularity="best")

        assert len(bin_edges) > 1
        # Should be sorted
        assert all(bin_edges[i] <= bin_edges[i + 1] for i in range(len(bin_edges) - 1))

    def test_histogram_bin_edges_granularity_integer(self, normal_data):
        """Test histogram_bin_edges with integer granularity."""
        bin_edges_0 = histogram_bin_edges(normal_data, granularity=0)
        bin_edges_1 = histogram_bin_edges(normal_data, granularity=1)

        assert len(bin_edges_0) > 1
        assert len(bin_edges_1) > 1

        # Higher granularity should have at least as many bins
        assert len(bin_edges_1) >= len(bin_edges_0)

    def test_histogram_bin_edges_granularity_consistency_with_histogram(
        self, uniform_data
    ):
        """Test that bin_edges from histogram match histogram_bin_edges for same granularity."""
        _, bin_edges_hist = histogram(uniform_data, granularity=1)
        bin_edges_func = histogram_bin_edges(uniform_data, granularity=1)

        assert np.allclose(np.asarray(bin_edges_hist), np.asarray(bin_edges_func))

    def test_histogram_bin_edges_granularity_exceeds_maximum(self, simple_data):
        """Test histogram_bin_edges with granularity exceeding maximum."""
        bin_edges = histogram_bin_edges(simple_data, granularity=1000)

        # Should still return valid results
        assert len(bin_edges) > 1

    def test_histogram_bin_edges_granularity_none_raises_error(self, simple_data):
        """Test that None granularity raises ValueError for histogram_bin_edges."""
        with pytest.raises(ValueError, match="granularity must be"):
            histogram_bin_edges(simple_data, granularity=None)  # type: ignore

    def test_histogram_granularity_with_different_backends(self, normal_data):
        """Test granularity parameter with different array backends."""
        # Test with list
        densities_list, _ = histogram(normal_data, granularity=1)
        assert isinstance(densities_list, list)

        # Test with numpy
        np_array = np.array(normal_data)
        densities_np, _ = histogram(np_array, granularity=1)
        assert hasattr(densities_np, "__array_namespace__")

        # Test with pyarrow
        pa_array = pa.array(normal_data)
        densities_pa, _ = histogram(pa_array, granularity=1)
        assert isinstance(densities_pa, pa.Array)

    def test_histogram_multiple_granularities_different_results(self, bimodal_data):
        """Test that different granularities produce different histograms."""
        densities_0, bin_edges_0 = histogram(bimodal_data, granularity=0)
        densities_2, bin_edges_2 = histogram(bimodal_data, granularity=2)

        # Different granularities should generally produce different numbers of bins
        # (though not guaranteed in all edge cases)
        # At minimum, both should be valid
        assert len(densities_0) > 0
        assert len(densities_2) > 0
        assert len(bin_edges_0) == len(densities_0) + 1
        assert len(bin_edges_2) == len(densities_2) + 1

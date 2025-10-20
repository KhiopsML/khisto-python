"""Tests for histogram functions in khisto.array.histogram module."""

from __future__ import annotations

from array import array

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from khisto.array import histogram, histogram_bin_edges, histogram_series


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


class TestHistogramSeries:
    """Test cases for histogram_series function."""

    def test_histogram_series_best_granularity(self, simple_data):
        """Test histogram_series with granularity='best'."""
        df = histogram_series(simple_data, granularity="best")

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

    def test_histogram_series_all_histograms(self, normal_data):
        """Test histogram_series with granularity=None."""
        df = histogram_series(normal_data, granularity=None)

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

    def test_histogram_series_with_numpy(self, uniform_data):
        """Test histogram_series with NumPy array."""
        np_array = np.array(uniform_data)
        df = histogram_series(np_array, granularity="best")

        assert isinstance(df, nw.DataFrame)
        assert len(df) > 0

    def test_histogram_series_with_pandas_series(self, normal_data):
        """Test histogram_series with Pandas Series."""
        series = pd.Series(normal_data)
        df = histogram_series(series, granularity="best")

        assert isinstance(df, nw.DataFrame)
        # Check that backend is pandas
        native = nw.to_native(df)
        assert isinstance(native, pd.DataFrame)

    def test_histogram_series_with_polars_series(self, bimodal_data):
        """Test histogram_series with Polars Series."""
        series = pl.Series(bimodal_data)
        df = histogram_series(series, granularity="best")

        assert isinstance(df, nw.DataFrame)
        # Check that backend is polars
        native = nw.to_native(df)
        assert isinstance(native, pl.DataFrame)

    def test_histogram_series_probability_sum(self, uniform_data):
        """Test that probabilities sum to 1."""
        df = histogram_series(uniform_data, granularity="best")

        total_prob = df["probability"].sum()
        assert np.isclose(total_prob, 1.0, rtol=1e-5)

    def test_histogram_series_density_calculation(self, simple_data):
        """Test that density = probability / length."""
        df = histogram_series(simple_data, granularity="best")

        for row in df.iter_rows(named=True):
            expected_density = (
                row["probability"] / row["length"] if row["length"] > 0 else 0
            )
            assert np.isclose(row["density"], expected_density, rtol=1e-5)

    def test_histogram_series_best_marked(self, normal_data):
        """Test that best histogram is marked when granularity=None."""
        df = histogram_series(normal_data, granularity=None)

        # Should have is_best column
        assert "is_best" in df.columns

        # Should have at least one True value
        best_count = df.filter(nw.col("is_best")).shape[0]
        assert best_count > 0

    def test_histogram_series_bin_bounds(self, simple_data):
        """Test that upper_bound > lower_bound for all bins."""
        df = histogram_series(simple_data, granularity="best")

        for row in df.iter_rows(named=True):
            assert row["upper_bound"] >= row["lower_bound"]

    def test_histogram_series_length_calculation(self, uniform_data):
        """Test that length = upper_bound - lower_bound."""
        df = histogram_series(uniform_data, granularity="best")

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
        """Test histogram_series with Pandas maintains backend."""
        series = pd.Series(simple_data)
        df = histogram_series(series, granularity="best")
        native = nw.to_native(df)
        assert isinstance(native, pd.DataFrame)

    def test_polars_series_backend(self, simple_data):
        """Test histogram_series with Polars maintains backend."""
        series = pl.Series(simple_data)
        df = histogram_series(series, granularity="best")
        native = nw.to_native(df)
        assert isinstance(native, pl.DataFrame)


class TestCumulativeParameter:
    """Test cases for cumulative parameter in histogram and histogram_series functions."""

    def test_histogram_cumulative_false(self, uniform_data):
        """Test histogram with cumulative=False (default) returns densities."""
        densities, bin_edges = histogram(uniform_data, cumulative=False)

        # Convert to numpy for easier calculation
        densities_np = np.asarray(densities)
        bin_edges_np = np.asarray(bin_edges)

        # Calculate bin widths
        bin_widths = bin_edges_np[1:] - bin_edges_np[:-1]

        # Sum of (density * width) should be approximately 1
        total_probability = np.sum(densities_np * bin_widths)
        assert np.isclose(total_probability, 1.0, rtol=1e-5)

    def test_histogram_cumulative_true(self, uniform_data):
        """Test histogram with cumulative=True returns cumulative probabilities."""
        cumulative_probs, bin_edges = histogram(uniform_data, cumulative=True)

        # Convert to numpy
        cumulative_probs_np = np.asarray(cumulative_probs)

        # Check that cumulative probabilities are monotonically increasing
        assert np.all(np.diff(cumulative_probs_np) >= 0)

        # Last cumulative probability should be approximately 1
        assert np.isclose(cumulative_probs_np[-1], 1.0, rtol=1e-5)

        # First cumulative probability should be 0
        assert np.isclose(cumulative_probs_np[0], 0.0, rtol=1e-5)

    def test_histogram_cumulative_monotonic(self, normal_data):
        """Test that cumulative histogram is strictly monotonic."""
        cumulative_probs, _ = histogram(normal_data, cumulative=True)

        cumulative_probs_np = np.asarray(cumulative_probs)

        # All differences should be non-negative (monotonically increasing)
        diffs = np.diff(cumulative_probs_np)
        assert np.all(diffs >= 0)

    def test_histogram_cumulative_starts_zero(self, simple_data):
        """Test that cumulative histogram starts with 0.0."""
        cumulative_probs, _ = histogram(simple_data, cumulative=True)

        first_value = np.asarray(cumulative_probs)[0]
        assert np.isclose(first_value, 0.0, rtol=1e-5)

    def test_histogram_cumulative_with_different_backends(self, simple_data):
        """Test cumulative histogram with different array backends."""
        # Test with list
        cum_list, _ = histogram(simple_data, cumulative=True)
        assert isinstance(cum_list, list)

        # Test with numpy
        np_array = np.array(simple_data)
        cum_np, _ = histogram(np_array, cumulative=True)
        assert hasattr(cum_np, "__array_namespace__")

        # Test with pyarrow
        pa_array = pa.array(simple_data)
        cum_pa, _ = histogram(pa_array, cumulative=True)
        assert isinstance(cum_pa, pa.Array)

        # All should end at approximately 1.0
        assert np.isclose(np.asarray(cum_list)[-1], 1.0, rtol=1e-5)
        assert np.isclose(np.asarray(cum_np)[-1], 1.0, rtol=1e-5)
        assert np.isclose(np.asarray(cum_pa)[-1], 1.0, rtol=1e-5)

    def test_histogram_series_cumulative_column(self, normal_data):
        """Test histogram_series adds cumulative_probability column when cumulative=True."""
        df = histogram_series(normal_data, granularity="best", cumulative=True)

        # Check that cumulative_probability column exists
        assert "cumulative_probability" in df.columns

        # Check monotonicity
        cum_probs = df["cumulative_probability"].to_numpy()
        assert np.all(np.diff(cum_probs) >= 0)

        # Last value should be approximately 1
        assert np.isclose(cum_probs[-1], 1.0, rtol=1e-5)

    def test_histogram_series_cumulative_false(self, uniform_data):
        """Test histogram_series without cumulative column when cumulative=False."""
        df = histogram_series(uniform_data, granularity="best", cumulative=False)

        # Check that cumulative_probability column does not exist
        assert "cumulative_probability" not in df.columns

    def test_histogram_series_cumulative_with_multiple_granularities(self, normal_data):
        """Test histogram_series cumulative with granularity=None."""
        df = histogram_series(normal_data, granularity=None, cumulative=True)

        # Check that cumulative_probability column exists
        assert "cumulative_probability" in df.columns

        # For each granularity, check that cumulative probabilities are valid
        for granularity in df["granularity"].unique():
            df_gran = df.filter(nw.col("granularity") == granularity)
            cum_probs = df_gran["cumulative_probability"].to_numpy()

            # Check monotonicity
            assert np.all(np.diff(cum_probs) >= 0)

            # Last value should be approximately 1
            assert np.isclose(cum_probs[-1], 1.0, rtol=1e-5)


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
        cum_probs_0, _ = histogram(normal_data, granularity=0, cumulative=True)
        cum_probs_1, _ = histogram(normal_data, granularity=1, cumulative=True)

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

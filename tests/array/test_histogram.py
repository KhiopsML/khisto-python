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

    def test_histogram_series_only_best(self, simple_data):
        """Test histogram_series with only_best=True."""
        df = histogram_series(simple_data, only_best=True)

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
        """Test histogram_series with only_best=False."""
        df = histogram_series(normal_data, only_best=False)

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
        df = histogram_series(np_array, only_best=True)

        assert isinstance(df, nw.DataFrame)
        assert len(df) > 0

    def test_histogram_series_with_pandas_series(self, normal_data):
        """Test histogram_series with Pandas Series."""
        series = pd.Series(normal_data)
        df = histogram_series(series, only_best=True)

        assert isinstance(df, nw.DataFrame)
        # Check that backend is pandas
        native = nw.to_native(df)
        assert isinstance(native, pd.DataFrame)

    def test_histogram_series_with_polars_series(self, bimodal_data):
        """Test histogram_series with Polars Series."""
        series = pl.Series(bimodal_data)
        df = histogram_series(series, only_best=True)

        assert isinstance(df, nw.DataFrame)
        # Check that backend is polars
        native = nw.to_native(df)
        assert isinstance(native, pl.DataFrame)

    def test_histogram_series_probability_sum(self, uniform_data):
        """Test that probabilities sum to 1."""
        df = histogram_series(uniform_data, only_best=True)

        total_prob = df["probability"].sum()
        assert np.isclose(total_prob, 1.0, rtol=1e-5)

    def test_histogram_series_density_calculation(self, simple_data):
        """Test that density = probability / length."""
        df = histogram_series(simple_data, only_best=True)

        for row in df.iter_rows(named=True):
            expected_density = (
                row["probability"] / row["length"] if row["length"] > 0 else 0
            )
            assert np.isclose(row["density"], expected_density, rtol=1e-5)

    def test_histogram_series_best_marked(self, normal_data):
        """Test that best histogram is marked when only_best=False."""
        df = histogram_series(normal_data, only_best=False)

        # Should have is_best column
        assert "is_best" in df.columns

        # Should have at least one True value
        best_count = df.filter(nw.col("is_best")).shape[0]
        assert best_count > 0

    def test_histogram_series_bin_bounds(self, simple_data):
        """Test that upper_bound > lower_bound for all bins."""
        df = histogram_series(simple_data, only_best=True)

        for row in df.iter_rows(named=True):
            assert row["upper_bound"] >= row["lower_bound"]

    def test_histogram_series_length_calculation(self, uniform_data):
        """Test that length = upper_bound - lower_bound."""
        df = histogram_series(uniform_data, only_best=True)

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
        df = histogram_series(series, only_best=True)
        native = nw.to_native(df)
        assert isinstance(native, pd.DataFrame)

    def test_polars_series_backend(self, simple_data):
        """Test histogram_series with Polars maintains backend."""
        series = pl.Series(simple_data)
        df = histogram_series(series, only_best=True)
        native = nw.to_native(df)
        assert isinstance(native, pl.DataFrame)

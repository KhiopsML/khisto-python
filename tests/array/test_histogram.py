"""Tests for histogram functions in khisto.array.histogram module."""

from __future__ import annotations

import numpy as np
import pytest

from khisto.array import histogram


# Test data fixtures
@pytest.fixture
def simple_data():
    """Simple test data: [1, 2, 3, 4, 5]"""
    return [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.fixture
def normal_data():
    """Normal distribution data"""
    np.random.seed(42)
    return np.random.normal(loc=0, scale=1, size=1000)


@pytest.fixture
def uniform_data():
    """Uniform distribution data"""
    np.random.seed(42)
    return np.random.uniform(low=0, high=10, size=500)


@pytest.fixture
def bimodal_data():
    """Bimodal distribution data"""
    np.random.seed(42)
    data1 = np.random.normal(loc=-2, scale=0.5, size=250)
    data2 = np.random.normal(loc=2, scale=0.5, size=250)
    return np.concatenate([data1, data2])


class TestHistogram:
    """Test cases for histogram function."""

    def test_histogram_with_list(self, simple_data):
        """Test histogram with Python list input."""
        hist, bin_edges = histogram(simple_data)

        assert isinstance(hist, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)
        assert len(hist) > 0
        assert len(bin_edges) == len(hist) + 1
        # Check bin edges are sorted
        assert np.all(np.diff(bin_edges) >= 0)

    def test_histogram_with_tuple(self, simple_data):
        """Test histogram with tuple input."""
        hist, bin_edges = histogram(tuple(simple_data))

        assert isinstance(hist, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)
        assert len(hist) > 0

    def test_histogram_with_numpy(self, normal_data):
        """Test histogram with NumPy array input."""
        hist, bin_edges = histogram(normal_data)

        assert isinstance(hist, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)
        assert len(hist) > 0
        assert len(bin_edges) == len(hist) + 1

    def test_histogram_density_normalization(self, uniform_data):
        """Test that histogram densities integrate to 1."""
        hist, bin_edges = histogram(uniform_data, density=True)

        # Calculate bin widths
        bin_widths = np.diff(bin_edges)

        # Sum of (density * width) should be approximately 1
        total_probability = np.sum(hist * bin_widths)
        assert np.isclose(total_probability, 1.0, rtol=1e-5)

    def test_histogram_frequency_sum(self, normal_data):
        """Test that histogram frequencies sum to total count."""
        hist, bin_edges = histogram(normal_data, density=False)

        # Sum of frequencies should equal total count
        assert np.sum(hist) == len(normal_data)

    def test_histogram_bin_coverage(self, simple_data):
        """Test that bin edges cover the full range of data."""
        hist, bin_edges = histogram(simple_data)

        assert bin_edges[0] <= min(simple_data)
        assert bin_edges[-1] >= max(simple_data)

    def test_histogram_with_duplicates(self):
        """Test histogram with duplicate values."""
        data = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        hist, bin_edges = histogram(data)

        assert len(hist) > 0
        assert len(bin_edges) == len(hist) + 1

    def test_histogram_with_range(self, normal_data):
        """Test histogram with range parameter."""
        hist, bin_edges = histogram(normal_data, range=(-1, 1))

        # Bin edges should be within the specified range
        assert bin_edges[0] >= -1
        assert bin_edges[-1] <= 1

    def test_histogram_with_max_bins(self, normal_data):
        """Test histogram with max_bins parameter."""
        hist, bin_edges = histogram(normal_data, max_bins=5)

        # Number of bins should not exceed max_bins
        assert len(hist) <= 5

    def test_histogram_density_parameter(self, normal_data):
        """Test density parameter behavior."""
        # Without density (counts)
        hist_counts, edges1 = histogram(normal_data, density=False)

        # With density
        hist_density, edges2 = histogram(normal_data, density=True)

        # Both should have same bin edges
        np.testing.assert_array_equal(edges1, edges2)

        # Counts should be integers (as floats)
        assert np.all(hist_counts == hist_counts.astype(int))

    def test_histogram_empty_after_range(self):
        """Test that empty array after range filtering raises ValueError."""
        data = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="empty"):
            histogram(data, range=(10, 20))


class TestHistogramNumpyCompatibility:
    """Test numpy.histogram compatibility."""

    def test_return_type(self, normal_data):
        """Test that return type matches numpy.histogram."""
        hist, bin_edges = histogram(normal_data)

        # Should return tuple of two ndarrays
        assert isinstance(hist, np.ndarray)
        assert isinstance(bin_edges, np.ndarray)

    def test_bin_edges_length(self, normal_data):
        """Test that bin_edges has length hist + 1."""
        hist, bin_edges = histogram(normal_data)
        assert len(bin_edges) == len(hist) + 1

    def test_2d_array_flattening(self):
        """Test that 2D arrays are flattened."""
        data_2d = np.array([[1, 2, 3], [4, 5, 6]])
        hist, bin_edges = histogram(data_2d)

        # Should process all 6 values
        assert np.sum(hist) == 6

# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests for matplotlib histogram plotting."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from khisto import histogram
from khisto.matplotlib import hist


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure, including when a test fails."""
    yield
    plt.close("all")


class TestHistBasic:
    """Test basic hist functionality."""

    @pytest.fixture
    def normal_data(self):
        """Normal distribution data"""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)

    def test_simple_array(self, normal_data):
        """Test hist with simple numpy array."""
        _fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, ax=ax)

        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert patches is not None
        assert len(n) > 0
        assert len(bins) == len(n) + 1

    def test_without_ax(self, normal_data):
        """Test hist without explicit ax parameter."""
        n, bins, _ = hist(normal_data)

        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert len(n) > 0

    def test_density_histogram(self, normal_data):
        """Test density histogram."""
        _fig, ax = plt.subplots()
        n, bins, _ = hist(normal_data, density=True, ax=ax)

        # Density should integrate to 1
        bin_widths = np.diff(bins)
        total = np.sum(n * bin_widths)
        assert np.isclose(total, 1.0, rtol=1e-5)

    def test_frequency_histogram(self, normal_data):
        """Test frequency histogram is the default behavior."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, ax=ax, density=False)

        # Frequencies should sum to total count
        assert np.sum(n) == len(normal_data)

    @pytest.mark.parametrize("density", [False, True])
    def test_histogram_matches_khiops_at_internal_edges(self, density):
        """Test that observations on internal edges keep their Khiops bins."""
        data = np.repeat([1, 2, 3, 4, 5, 6], [458, 82, 43, 11, 3, 2])
        expected, expected_bins = histogram(data, max_bins=100, density=density)
        _fig, ax = plt.subplots()

        values, bins, _ = hist(data, max_bins=100, density=density, ax=ax)

        np.testing.assert_array_equal(bins, expected_bins)
        np.testing.assert_allclose(values, expected)

    def test_horizontal_orientation(self, normal_data):
        """Test horizontal histogram."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, orientation="horizontal", ax=ax)

        assert isinstance(n, np.ndarray)
        assert len(n) > 0

    def test_with_max_bins(self, normal_data):
        """Test histogram with max_bins parameter."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, max_bins=5, ax=ax)

        assert len(n) <= 5

    def test_with_range(self, normal_data):
        """Test histogram with range parameter."""
        _fig, ax = plt.subplots()
        _, bins, _ = hist(normal_data, range=(-1, 1), ax=ax)

        assert bins[0] >= -1
        assert bins[-1] <= 1

    def test_log_scale(self, normal_data):
        """Test histogram with log scale."""
        _fig, ax = plt.subplots()
        hist(normal_data, log=True, ax=ax)

        assert ax.get_yscale() == "log"

    def test_color_parameter(self, normal_data):
        """Test histogram with color parameter."""
        _fig, ax = plt.subplots()
        _, _, patches = hist(normal_data, color="red", ax=ax)

        assert patches is not None

    def test_step_histtype(self, normal_data):
        """Test histogram with step histtype."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, histtype="step", ax=ax)

        assert isinstance(n, np.ndarray)

    def test_stepfilled_histtype(self, normal_data):
        """Test histogram with stepfilled histtype."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, histtype="stepfilled", ax=ax)

        assert isinstance(n, np.ndarray)

    def test_cumulative_density_histogram(self, normal_data):
        """Test cumulative density histogram."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, density=True, cumulative=True, ax=ax)

        assert np.isclose(n[-1], 1.0, rtol=1e-5)

    def test_cumulative_hist_matches_array_api(self, normal_data):
        """Test that the plotting wrapper matches cumulative histogram values."""
        _fig, ax = plt.subplots()
        n, bins, _ = hist(normal_data, density=True, cumulative=True, ax=ax)
        density, expected_bins = histogram(normal_data, density=True)
        expected = np.cumsum(density * np.diff(expected_bins))

        np.testing.assert_array_equal(bins, expected_bins)
        np.testing.assert_allclose(n, expected)

    def test_reverse_cumulative_frequency_histogram(self, normal_data):
        """Test reverse cumulative frequency histogram."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, density=False, cumulative=-1, ax=ax)

        assert np.isclose(n[0], len(normal_data))

    def test_unsupported_bins_parameter(self, normal_data):
        """Test that bins raises a clear error message."""
        _fig, ax = plt.subplots()

        with pytest.raises(TypeError, match="bins is not supported"):
            hist(normal_data, bins=10, ax=ax)


class TestHistReturnValues:
    """Test return values match matplotlib.pyplot.hist interface."""

    @pytest.fixture
    def data(self):
        np.random.seed(42)
        return np.random.normal(0, 1, 500)

    def test_return_tuple_structure(self, data):
        """Test that return is (n, bins, patches) tuple."""
        result = hist(data)

        assert isinstance(result, tuple)
        assert len(result) == 3

        n, bins, _ = result
        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)

    def test_bins_edges_count(self, data):
        """Test that bins has n+1 edges."""
        n, bins, _ = hist(data)
        assert len(bins) == len(n) + 1

# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests for matplotlib histogram plotting."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from khisto.matplotlib import hist
from khisto.array import histogram


class TestHistBasic:
    """Test basic hist functionality."""

    @pytest.fixture
    def normal_data(self):
        """Normal distribution data"""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)

    def test_simple_array(self, normal_data):
        """Test hist with simple numpy array."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, ax=ax)

        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert patches is not None
        assert len(n) > 0
        assert len(bins) == len(n) + 1
        plt.close(fig)

    def test_without_ax(self, normal_data):
        """Test hist without explicit ax parameter."""
        n, bins, patches = hist(normal_data)

        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert len(n) > 0
        plt.close()

    def test_density_histogram(self, normal_data):
        """Test density histogram."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, density=True, ax=ax)

        # Density should integrate to 1
        bin_widths = np.diff(bins)
        total = np.sum(n * bin_widths)
        assert np.isclose(total, 1.0, rtol=1e-5)
        plt.close(fig)

    def test_frequency_histogram(self, normal_data):
        """Test frequency histogram is the default behavior."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, ax=ax)

        # Frequencies should sum to total count
        assert np.sum(n) == len(normal_data)
        plt.close(fig)

    def test_horizontal_orientation(self, normal_data):
        """Test horizontal histogram."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, orientation="horizontal", ax=ax)

        assert isinstance(n, np.ndarray)
        assert len(n) > 0
        plt.close(fig)

    def test_with_max_bins(self, normal_data):
        """Test histogram with max_bins parameter."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, max_bins=5, ax=ax)

        assert len(n) <= 5
        plt.close(fig)

    def test_with_range(self, normal_data):
        """Test histogram with range parameter."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, range=(-1, 1), ax=ax)

        assert bins[0] >= -1
        assert bins[-1] <= 1
        plt.close(fig)

    def test_log_scale(self, normal_data):
        """Test histogram with log scale."""
        fig, ax = plt.subplots()
        hist(normal_data, log=True, ax=ax)

        assert ax.get_yscale() == "log"
        plt.close(fig)

    def test_color_parameter(self, normal_data):
        """Test histogram with color parameter."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, color="red", ax=ax)

        assert patches is not None
        plt.close(fig)

    def test_step_histtype(self, normal_data):
        """Test histogram with step histtype."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, histtype="step", ax=ax)

        assert isinstance(n, np.ndarray)
        plt.close(fig)

    def test_stepfilled_histtype(self, normal_data):
        """Test histogram with stepfilled histtype."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, histtype="stepfilled", ax=ax)

        assert isinstance(n, np.ndarray)
        plt.close(fig)

    def test_cumulative_density_histogram(self, normal_data):
        """Test cumulative density histogram."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, density=True, cumulative=True, ax=ax)

        assert np.isclose(n[-1], 1.0, rtol=1e-5)
        plt.close(fig)

    def test_cumulative_hist_matches_array_api(self, normal_data):
        """Test that the plotting wrapper matches cumulative histogram values."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, density=True, cumulative=True, ax=ax)
        density, expected_bins = histogram(normal_data, density=True)
        expected = np.cumsum(density * np.diff(expected_bins))

        np.testing.assert_array_equal(bins, expected_bins)
        np.testing.assert_allclose(n, expected)
        plt.close(fig)

    def test_reverse_cumulative_frequency_histogram(self, normal_data):
        """Test reverse cumulative frequency histogram."""
        fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, cumulative=-1, ax=ax)

        assert np.isclose(n[0], len(normal_data))
        plt.close(fig)

    def test_sequence_of_arrays_is_combined(self, normal_data):
        """Test that sequences of arrays are combined into one histogram."""
        fig, ax = plt.subplots()
        datasets = [normal_data[:500], normal_data[500:]]
        n, bins, patches = hist(datasets, ax=ax)

        assert np.sum(n) == len(normal_data)
        plt.close(fig)

    def test_unsupported_bins_parameter(self, normal_data):
        """Test that bins raises a clear error message."""
        fig, ax = plt.subplots()

        with pytest.raises(TypeError, match="bins is not supported"):
            hist(normal_data, bins=10, ax=ax)

        plt.close(fig)


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

        n, bins, patches = result
        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)

    def test_bins_edges_count(self, data):
        """Test that bins has n+1 edges."""
        n, bins, patches = hist(data)
        assert len(bins) == len(n) + 1

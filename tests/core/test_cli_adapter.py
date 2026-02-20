# Copyright (c) 2023-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests for khisto.core.cli_adapter module."""

import pathlib
import numpy as np
import pytest

from khisto.core.cli_adapter import (
    _parse_file_type,
    HistogramResult,
)
from khisto.core import compute_histogram


class TestParseFileType:
    """Tests for _parse_file_type function."""

    def test_best_histogram_file(self):
        """Test parsing histogram.csv file."""
        file_path = pathlib.Path("histogram.csv")
        ftype, hist_id = _parse_file_type(file_path)
        assert ftype == "best_histogram"
        assert hist_id is None

    def test_series_file(self):
        """Test parsing histogram.series.csv file."""
        file_path = pathlib.Path("histogram.series.csv")
        ftype, hist_id = _parse_file_type(file_path)
        assert ftype == "series"
        assert hist_id is None

    def test_numbered_histogram_file(self):
        """Test parsing histogram.N.csv files."""
        file_path = pathlib.Path("histogram.1.csv")
        ftype, hist_id = _parse_file_type(file_path)
        assert ftype == "histogram"
        assert hist_id == 0

        file_path = pathlib.Path("histogram.5.csv")
        ftype, hist_id = _parse_file_type(file_path)
        assert ftype == "histogram"
        assert hist_id == 4

        file_path = pathlib.Path("histogram.123.csv")
        ftype, hist_id = _parse_file_type(file_path)
        assert ftype == "histogram"
        assert hist_id == 122

    def test_invalid_file_name(self):
        """Test that invalid file names raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized histogram file name"):
            _parse_file_type(pathlib.Path("invalid.csv"))

        with pytest.raises(ValueError, match="Unrecognized histogram file name"):
            _parse_file_type(pathlib.Path("histogram.txt"))

        with pytest.raises(ValueError, match="Unrecognized histogram file name"):
            _parse_file_type(pathlib.Path("histogram.abc.csv"))


class TestHistogramResult:
    """Tests for HistogramResult dataclass."""

    def test_histogram_result_properties(self):
        """Test HistogramResult properties."""
        result = HistogramResult(
            lower_bound=np.array([0.0, 1.0, 2.0]),
            upper_bound=np.array([1.0, 2.0, 3.0]),
            frequency=np.array([10, 20, 15]),
            probability=np.array([0.222, 0.444, 0.333]),
            density=np.array([0.222, 0.444, 0.333]),
            is_best=True,
            granularity=2,
        )

        # Test bin_edges property
        expected_edges = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.bin_edges, expected_edges)

        # Test bin_widths property
        expected_widths = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result.bin_widths, expected_widths)

        # Test bin_centers property
        expected_centers = np.array([0.5, 1.5, 2.5])
        np.testing.assert_array_equal(result.bin_centers, expected_centers)

        # Test __len__
        assert len(result) == 3

        # Test is_best and granularity
        assert result.is_best is True
        assert result.granularity == 2


class TestComputeHistogram:
    """Tests for compute_histogram function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)

    def test_compute_histogram_basic(self, sample_data):
        """Test basic histogram computation."""
        results = compute_histogram(sample_data)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, HistogramResult) for r in results)

        # Check each result has valid structure
        for result in results:
            assert len(result) > 0
            assert len(result.bin_edges) == len(result) + 1

    def test_compute_histogram_returns_all_granularities(self, sample_data):
        """Test histogram returning all granularities."""
        results = compute_histogram(sample_data)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, HistogramResult) for r in results)

        # Exactly one should be marked as best
        best_count = sum(1 for r in results if r.is_best)
        assert best_count == 1

        # Granularities should be sorted
        granularities = [r.granularity for r in results]
        assert granularities == sorted(granularities)

    def test_compute_histogram_empty_input(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_histogram(np.array([]))

    def test_compute_histogram_nan_handling(self):
        """Test that NaN values are filtered out."""
        data = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
        results = compute_histogram(data)

        # Should process 4 valid values - check the finest granularity result
        finest_result = results[-1]
        assert np.sum(finest_result.frequency) == 4

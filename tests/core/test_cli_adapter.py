# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests for khisto.core.cli_adapter module."""

import csv
import pathlib
import subprocess
import numpy as np
import pytest

from khisto.core.cli_adapter import (
    _parse_file_type,
    _process_histogram_files,
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

    def test_compute_histogram_reports_cli_failure(self, monkeypatch):
        """Test that CLI failures raise a descriptive runtime error."""

        def fail_run(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["khisto", "-b", "-e", "input.bin", "histogram.csv"],
                output="khisto stdout message",
                stderr="khisto stderr message",
            )

        monkeypatch.setattr("khisto.core.cli_adapter.subprocess.run", fail_run)

        with pytest.raises(RuntimeError, match="exit code 1") as excinfo:
            compute_histogram(np.array([0.0, 1.0]))

        message = str(excinfo.value)
        assert "khisto stdout message" in message
        assert "khisto stderr message" in message

    def test_compute_histogram_writes_binary_input(self, monkeypatch):
        """Test that compute_histogram always writes a binary input file."""

        expected_result = [
            HistogramResult(
                lower_bound=np.array([0.0]),
                upper_bound=np.array([1.0]),
                frequency=np.array([2]),
                probability=np.array([1.0]),
                density=np.array([1.0]),
                is_best=True,
                granularity=0,
            )
        ]
        captured_values = None

        monkeypatch.setattr(
            "khisto.core.cli_adapter._process_histogram_files",
            lambda temp_dir, base_name: expected_result,
        )

        def fake_run(cmd, **kwargs):
            nonlocal captured_values
            assert "-b" in cmd
            input_path = pathlib.Path(cmd[-2])
            captured_values = np.fromfile(input_path, dtype=np.float64)

        monkeypatch.setattr("khisto.core.cli_adapter.subprocess.run", fake_run)

        results = compute_histogram(np.array([0.25, 0.75]))

        assert results == expected_result
        np.testing.assert_array_equal(captured_values, np.array([0.25, 0.75]))


class TestProcessHistogramFiles:
    """Tests for histogram file post-processing."""

    @staticmethod
    def _write_csv(path: pathlib.Path, header: list[str], rows: list[list[object]]):
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def test_series_raw_marks_finest_interpretable_histogram(self, tmp_path):
        """Test that Raw=1 on the finest histogram selects the previous histogram."""
        histogram_header = [
            "LowerBound",
            "UpperBound",
            "Length",
            "Frequency",
            "Probability",
            "Density",
        ]
        series_header = [
            "FileName",
            "Granularity",
            "IntervalNumber",
            "PeakIntervalNumber",
            "SpikeIntervalNumber",
            "EmptyIntervalNumber",
            "Level",
            "InformationRate",
            "TruncationEpsilon",
            "RemovedSingularityNumber",
            "Raw",
        ]

        self._write_csv(
            tmp_path / "histogram.1.csv",
            histogram_header,
            [[0.0, 3.0, 3.0, 3, 1.0, 1.0 / 3.0]],
        )
        self._write_csv(
            tmp_path / "histogram.2.csv",
            histogram_header,
            [
                [0.0, 1.0, 1.0, 1, 1.0 / 3.0, 1.0 / 3.0],
                [1.0, 3.0, 2.0, 2, 2.0 / 3.0, 1.0 / 3.0],
            ],
        )
        self._write_csv(
            tmp_path / "histogram.csv",
            histogram_header,
            [
                [0.0, 1.0, 1.0, 1, 1.0 / 3.0, 1.0 / 3.0],
                [1.0, 3.0, 2.0, 2, 2.0 / 3.0, 1.0 / 3.0],
            ],
        )
        self._write_csv(
            tmp_path / "histogram.series.csv",
            series_header,
            [
                [
                    "exploratory_analysis\\adult_age\\histogram.1.csv",
                    0,
                    1,
                    0,
                    0,
                    0,
                    10.0,
                    100.0,
                    0.0,
                    0,
                    0.0,
                ],
                [
                    "exploratory_analysis\\adult_age\\histogram.2.csv",
                    2,
                    2,
                    0,
                    0,
                    0,
                    1.0,
                    10.0,
                    0.0,
                    0,
                    1.0,
                ],
            ],
        )

        results = _process_histogram_files(str(tmp_path), "histogram")

        assert len(results) == 2
        assert [result.is_best for result in results] == [True, False]

    def test_series_raw_keeps_finest_histogram_when_interpretable(self, tmp_path):
        """Test that the finest histogram is selected when Raw=0."""
        histogram_header = [
            "LowerBound",
            "UpperBound",
            "Length",
            "Frequency",
            "Probability",
            "Density",
        ]
        series_header = [
            "FileName",
            "Granularity",
            "IntervalNumber",
            "PeakIntervalNumber",
            "SpikeIntervalNumber",
            "EmptyIntervalNumber",
            "Level",
            "InformationRate",
            "TruncationEpsilon",
            "RemovedSingularityNumber",
            "Raw",
        ]

        self._write_csv(
            tmp_path / "histogram.1.csv",
            histogram_header,
            [[0.0, 3.0, 3.0, 3, 1.0, 1.0 / 3.0]],
        )
        self._write_csv(
            tmp_path / "histogram.2.csv",
            histogram_header,
            [
                [0.0, 1.0, 1.0, 1, 1.0 / 3.0, 1.0 / 3.0],
                [1.0, 3.0, 2.0, 2, 2.0 / 3.0, 1.0 / 3.0],
            ],
        )
        self._write_csv(
            tmp_path / "histogram.series.csv",
            series_header,
            [
                [
                    str(tmp_path / "histogram.1.csv"),
                    0,
                    1,
                    0,
                    0,
                    0,
                    10.0,
                    100.0,
                    0.0,
                    0,
                    0.0,
                ],
                [
                    str(tmp_path / "histogram.2.csv"),
                    2,
                    2,
                    0,
                    0,
                    0,
                    1.0,
                    10.0,
                    0.0,
                    0,
                    0.0,
                ],
            ],
        )

        results = _process_histogram_files(str(tmp_path), "histogram")

        assert len(results) == 2
        assert [result.is_best for result in results] == [False, True]

    def test_best_histogram_file_is_fallback_without_series(self, tmp_path):
        """Test that histogram.csv remains the fallback when series is absent."""
        histogram_header = [
            "LowerBound",
            "UpperBound",
            "Length",
            "Frequency",
            "Probability",
            "Density",
        ]

        self._write_csv(
            tmp_path / "histogram.1.csv",
            histogram_header,
            [[0.0, 3.0, 3.0, 3, 1.0, 1.0 / 3.0]],
        )
        self._write_csv(
            tmp_path / "histogram.2.csv",
            histogram_header,
            [
                [0.0, 1.0, 1.0, 1, 1.0 / 3.0, 1.0 / 3.0],
                [1.0, 3.0, 2.0, 2, 2.0 / 3.0, 1.0 / 3.0],
            ],
        )
        self._write_csv(
            tmp_path / "histogram.csv",
            histogram_header,
            [
                [0.0, 1.0, 1.0, 1, 1.0 / 3.0, 1.0 / 3.0],
                [1.0, 3.0, 2.0, 2, 2.0 / 3.0, 1.0 / 3.0],
            ],
        )

        results = _process_histogram_files(str(tmp_path), "histogram")

        assert len(results) == 2
        assert [result.is_best for result in results] == [False, True]

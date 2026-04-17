# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests for khisto.core.backend module."""

import json
import pathlib
import subprocess

import numpy as np
import pytest

from khisto.core import compute_histograms
from khisto.core.backend import (
    _parse_file_type,
    _process_histogram_files,
    HistogramResult,
)


def _histogram_payload(
    lower_bounds: list[float],
    upper_bounds: list[float],
    frequencies: list[int],
) -> dict[str, list[float] | list[int]]:
    lengths = [upper - lower for lower, upper in zip(lower_bounds, upper_bounds)]
    total_frequency = float(sum(frequencies))
    probabilities = [frequency / total_frequency for frequency in frequencies]
    densities = [
        probability / length for probability, length in zip(probabilities, lengths)
    ]
    return {
        "lowerBounds": lower_bounds,
        "upperBounds": upper_bounds,
        "lengths": lengths,
        "frequencies": frequencies,
        "probabilities": probabilities,
        "densities": densities,
    }


class TestParseFileType:
    """Tests for _parse_file_type function."""

    def test_series_json_file(self):
        """Test validation of histogram.series.json file."""
        _parse_file_type(pathlib.Path("histogram.series.json"))

    def test_binary_series_json_file(self):
        """Test validation of histogram.series.bin.json file."""
        _parse_file_type(pathlib.Path("gaussian_histogram.series.bin.json"))

    def test_invalid_file_name(self):
        """Test that invalid file names raise ValueError."""
        with pytest.raises(ValueError, match="Unrecognized histogram file name"):
            _parse_file_type(pathlib.Path("histogram.json"))

        with pytest.raises(ValueError, match="Unrecognized histogram file name"):
            _parse_file_type(pathlib.Path("histogram.csv"))

        with pytest.raises(ValueError, match="Unrecognized histogram file name"):
            _parse_file_type(pathlib.Path("histogram.series.csv"))


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

        expected_edges = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result.bin_edges, expected_edges)

        expected_widths = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result.bin_widths, expected_widths)

        expected_centers = np.array([0.5, 1.5, 2.5])
        np.testing.assert_array_equal(result.bin_centers, expected_centers)

        assert len(result) == 3
        assert result.is_best is True
        assert result.granularity == 2


class TestComputeHistograms:
    """Tests for compute_histograms function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)

    def test_compute_histograms_basic(self, sample_data):
        """Test basic histogram computation."""
        results = compute_histograms(sample_data)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, HistogramResult) for r in results)

        for result in results:
            assert len(result) > 0
            assert len(result.bin_edges) == len(result) + 1

    def test_compute_histograms_returns_all_granularities(self, sample_data):
        """Test histogram returning all granularities."""
        results = compute_histograms(sample_data)

        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, HistogramResult) for r in results)

        best_count = sum(1 for r in results if r.is_best)
        assert best_count == 1

        granularities = [r.granularity for r in results]
        assert granularities == sorted(granularities)

    def test_compute_histograms_empty_input(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            compute_histograms(np.array([]))

    def test_compute_histograms_nan_handling(self):
        """Test that NaN values are filtered out."""
        data = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
        results = compute_histograms(data)

        finest_result = results[-1]
        assert np.sum(finest_result.frequency) == 4

    def test_compute_histograms_reports_cli_failure(self, monkeypatch):
        """Test that CLI failures raise a descriptive runtime error."""

        def fail_run(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=["khisto", "-b", "-e", "-j", "input.bin", "histogram.series.json"],
                output="khisto stdout message",
                stderr="khisto stderr message",
            )

        monkeypatch.setattr("khisto.core.backend.subprocess.run", fail_run)

        with pytest.raises(RuntimeError, match="exit code 1") as excinfo:
            compute_histograms(np.array([0.0, 1.0]))

        message = str(excinfo.value)
        assert "khisto stdout message" in message
        assert "khisto stderr message" in message
        assert "-j" in message

    def test_compute_histograms_writes_binary_input(self, monkeypatch):
        """Test that compute_histograms writes binary input and requests JSON output."""

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
            "khisto.core.backend._process_histogram_files",
            lambda temp_dir, base_name: expected_result,
        )

        def fake_run(cmd, **kwargs):
            nonlocal captured_values
            assert "-b" in cmd
            assert "-j" in cmd
            assert cmd[-1].endswith("histogram.series.json")
            input_path = pathlib.Path(cmd[-2])
            captured_values = np.fromfile(input_path, dtype=np.float64)

        monkeypatch.setattr("khisto.core.backend.subprocess.run", fake_run)

        results = compute_histograms(np.array([0.25, 0.75]))

        assert results == expected_result
        np.testing.assert_array_equal(captured_values, np.array([0.25, 0.75]))


class TestProcessHistogramFiles:
    """Tests for histogram JSON post-processing."""

    @staticmethod
    def _write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream)

    def test_series_selects_finest_interpretable_histogram(self, tmp_path):
        """Test that interpretableHistogramNumber selects the best histogram."""
        payload = {
            "tool": "Khiops Histogram",
            "version": "1.1",
            "bestHistogram": _histogram_payload([0.0, 1.0], [1.0, 3.0], [1, 2]),
            "histogramSeries": {
                "histogramNumber": 3,
                "interpretableHistogramNumber": 2,
                "truncationEpsilon": 0.0,
                "removedSingularIntervalNumber": 1,
                "granularities": [0, 2, 3],
                "intervalNumbers": [1, 2, 3],
                "peakIntervalNumbers": [0, 0, 0],
                "spikeIntervalNumbers": [0, 0, 0],
                "emptyIntervalNumbers": [0, 0, 0],
                "levels": [10.0, 15.0, 16.0],
                "informationRates": [66.0, 100.0, 101.0],
                "histograms": [
                    _histogram_payload([0.0], [3.0], [3]),
                    _histogram_payload([0.0, 1.0], [1.0, 3.0], [1, 2]),
                    _histogram_payload([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [1, 1, 1]),
                ],
            },
        }
        self._write_json(tmp_path / "histogram.series.json", payload)

        results = _process_histogram_files(str(tmp_path), "histogram.series")

        assert len(results) == 3
        assert [result.granularity for result in results] == [0, 2, 3]
        assert [result.is_best for result in results] == [False, True, False]

    def test_series_keeps_finest_histogram_when_all_interpretable(self, tmp_path):
        """Test that the finest histogram is selected when all are interpretable."""
        payload = {
            "tool": "Khiops Histogram",
            "version": "1.1",
            "bestHistogram": _histogram_payload([0.0, 1.0], [1.0, 3.0], [1, 2]),
            "histogramSeries": {
                "histogramNumber": 2,
                "interpretableHistogramNumber": 2,
                "truncationEpsilon": 0.0,
                "removedSingularIntervalNumber": 0,
                "granularities": [0, 2],
                "intervalNumbers": [1, 2],
                "peakIntervalNumbers": [0, 0],
                "spikeIntervalNumbers": [0, 0],
                "emptyIntervalNumbers": [0, 0],
                "levels": [10.0, 15.0],
                "informationRates": [66.0, 100.0],
                "histograms": [
                    _histogram_payload([0.0], [3.0], [3]),
                    _histogram_payload([0.0, 1.0], [1.0, 3.0], [1, 2]),
                ],
            },
        }
        self._write_json(tmp_path / "histogram.series.json", payload)

        results = _process_histogram_files(str(tmp_path), "histogram.series")

        assert len(results) == 2
        assert [result.is_best for result in results] == [False, True]

    def test_best_histogram_is_fallback_without_series(self, tmp_path):
        """Test that bestHistogram remains the fallback when the series is absent."""
        payload = {
            "tool": "Khiops Histogram",
            "version": "1.1",
            "bestHistogram": _histogram_payload([0.0, 1.0], [1.0, 3.0], [1, 2]),
        }
        self._write_json(tmp_path / "histogram.series.json", payload)

        results = _process_histogram_files(str(tmp_path), "histogram.series")

        assert len(results) == 1
        assert results[0].is_best is True
        assert results[0].granularity == 0

    def test_best_histogram_is_fallback_when_series_best_index_is_missing(
        self, tmp_path
    ):
        """Test that bestHistogram matching is used when series metadata is incomplete."""
        payload = {
            "tool": "Khiops Histogram",
            "version": "1.1",
            "bestHistogram": _histogram_payload([0.0, 1.0], [1.0, 3.0], [1, 2]),
            "histogramSeries": {
                "histogramNumber": 2,
                "granularities": [0, 2],
                "histograms": [
                    _histogram_payload([0.0], [3.0], [3]),
                    _histogram_payload([0.0, 1.0], [1.0, 3.0], [1, 2]),
                ],
            },
        }
        self._write_json(tmp_path / "histogram.series.json", payload)

        results = _process_histogram_files(str(tmp_path), "histogram.series")

        assert len(results) == 2
        assert [result.is_best for result in results] == [False, True]

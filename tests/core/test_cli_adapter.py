"""Tests for khisto.core.cli_adapter module."""

import pathlib
import subprocess
from unittest.mock import MagicMock, patch

import pyarrow as pa
from pyarrow import compute
import pytest

from khisto.core.cli_adapter import (
    _parse_file_type,
    _process_histogram_files,
    compute_histogram,
)


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


class TestHistogram:
    """Tests for histogram function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pa.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])

    @pytest.fixture
    def mock_subprocess_success(self):
        """Mock successful subprocess execution."""
        with patch("khisto.core.cli_adapter.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            yield mock_run

    def test_histogram_best_mode(self, sample_data, tmp_path, mock_subprocess_success):
        """Test histogram with granularity='best'."""
        # Create mock histogram file
        histogram_file = tmp_path / "histogram.csv"
        histogram_file.write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n"
            "1.0,3.0,2.0,5,0.5,0.25\n"
            "3.0,5.0,2.0,4,0.4,0.20\n"
        )

        with patch(
            "khisto.core.cli_adapter.tempfile.mkdtemp", return_value=str(tmp_path)
        ):
            with patch(
                "khisto.core.cli_adapter.tempfile.NamedTemporaryFile"
            ) as mock_temp:
                mock_file = MagicMock()
                mock_file.name = str(tmp_path / "input.txt")
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_temp.return_value = mock_file

                result = compute_histogram(sample_data, granularity="best")

                # Check columns exist
                expected_columns = {
                    "lower_bound",
                    "upper_bound",
                    "length",
                    "frequency",
                    "probability",
                    "density",
                    "granularity",
                    "is_best",
                }
                assert set(result.column_names) == expected_columns

                # Check all rows are marked as best
                assert all(result["is_best"].to_pylist())

                # Check granularity is 0
                assert all(g == 0 for g in result["granularity"].to_pylist())

    def test_histogram_all_granularities_mode(
        self, sample_data, tmp_path, mock_subprocess_success
    ):
        """Test histogram with granularity=None (all granularities)."""
        # Create mock histogram files
        (tmp_path / "histogram.1.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n1.0,5.0,4.0,9,1.0,0.25\n"
        )
        (tmp_path / "histogram.2.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n"
            "1.0,3.0,2.0,6,0.67,0.33\n"
            "3.0,5.0,2.0,3,0.33,0.17\n"
        )
        (tmp_path / "histogram.series.csv").write_text(
            "FileName,Granularity,IntervalNumber,PeakIntervalNumber,SpikeIntervalNumber,"
            "EmptyIntervalNumber,Level,InformationRate,TruncationEpsilon,"
            "RemovedSingularityNumber,Raw\n"
            "histogram,1,1,0,0,0,0.5,0.8,0.01,0,false\n"
            "histogram,2,2,1,0,0,0.9,0.95,0.01,0,false\n"
        )

        with patch(
            "khisto.core.cli_adapter.tempfile.mkdtemp", return_value=str(tmp_path)
        ):
            with patch(
                "khisto.core.cli_adapter.tempfile.NamedTemporaryFile"
            ) as mock_temp:
                mock_file = MagicMock()
                mock_file.name = str(tmp_path / "input.txt")
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_temp.return_value = mock_file

                result = compute_histogram(sample_data, granularity=None)

                # Check we have rows from both granularities
                granularities = set(result["granularity"].to_pylist())
                assert 0 in granularities
                assert 1 in granularities

                # Check that best histogram is marked (granularity 2 has highest level)
                best_rows = result.filter(result["is_best"])
                assert len(best_rows) > 0
                assert all(g == 1 for g in best_rows["granularity"].to_pylist())

    def test_histogram_command_construction(
        self, sample_data, tmp_path, mock_subprocess_success
    ):
        """Test that the correct command is constructed for khisto CLI."""
        # Test granularity='best' mode
        with patch(
            "khisto.core.cli_adapter.tempfile.mkdtemp", return_value=str(tmp_path)
        ):
            with patch(
                "khisto.core.cli_adapter.tempfile.NamedTemporaryFile"
            ) as mock_temp:
                mock_file = MagicMock()
                mock_file.name = str(tmp_path / "input.txt")
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_temp.return_value = mock_file

                # Create minimal histogram file to avoid errors
                (tmp_path / "histogram.csv").write_text(
                    "LowerBound,UpperBound,Length,Frequency,Probability,Density\n"
                    "1.0,5.0,4.0,9,1.0,0.25\n"
                )

                compute_histogram(sample_data, granularity="best")
                args = mock_subprocess_success.call_args[0][0]
                assert "-e" not in args

        mock_subprocess_success.reset_mock()

        # Test exploratory mode (granularity=None) with a fresh tmp directory
        tmp_path2 = tmp_path / "exploratory"
        tmp_path2.mkdir(parents=True, exist_ok=True)

        with patch(
            "khisto.core.cli_adapter.tempfile.mkdtemp", return_value=str(tmp_path2)
        ):
            with patch(
                "khisto.core.cli_adapter.tempfile.NamedTemporaryFile"
            ) as mock_temp:
                mock_file = MagicMock()
                mock_file.name = str(tmp_path2 / "input.txt")
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_temp.return_value = mock_file

                # Create histogram file for exploratory mode
                (tmp_path2 / "histogram.1.csv").write_text(
                    "LowerBound,UpperBound,Length,Frequency,Probability,Density\n"
                    "1.0,5.0,4.0,9,1.0,0.25\n"
                )

                compute_histogram(sample_data, granularity=None)
                args = mock_subprocess_success.call_args[0][0]
                assert "-e" in args

    def test_histogram_subprocess_error(self, sample_data, tmp_path):
        """Test that subprocess errors are properly handled."""
        with patch("khisto.core.cli_adapter.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "khisto", stderr="Error message"
            )

            with patch(
                "khisto.core.cli_adapter.tempfile.mkdtemp", return_value=str(tmp_path)
            ):
                with patch(
                    "khisto.core.cli_adapter.tempfile.NamedTemporaryFile"
                ) as mock_temp:
                    mock_file = MagicMock()
                    mock_file.name = str(tmp_path / "input.txt")
                    mock_file.__enter__ = MagicMock(return_value=mock_file)
                    mock_file.__exit__ = MagicMock(return_value=False)
                    mock_temp.return_value = mock_file

                    with pytest.raises(subprocess.CalledProcessError):
                        compute_histogram(sample_data, granularity="best")

    def test_histogram_specific_granularity(
        self, sample_data, tmp_path, mock_subprocess_success
    ):
        """Test histogram with a specific granularity level."""
        # Create mock histogram files with multiple granularities
        (tmp_path / "histogram.1.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n1.0,5.0,4.0,9,1.0,0.25\n"
        )
        (tmp_path / "histogram.2.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n"
            "1.0,3.0,2.0,6,0.67,0.33\n"
            "3.0,5.0,2.0,3,0.33,0.17\n"
        )
        (tmp_path / "histogram.3.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n"
            "1.0,2.0,1.0,3,0.33,0.33\n"
            "2.0,3.0,1.0,3,0.33,0.33\n"
            "3.0,5.0,2.0,3,0.33,0.17\n"
        )
        (tmp_path / "histogram.series.csv").write_text(
            "FileName,Granularity,IntervalNumber,PeakIntervalNumber,SpikeIntervalNumber,"
            "EmptyIntervalNumber,Level,InformationRate,TruncationEpsilon,"
            "RemovedSingularityNumber,Raw\n"
            "histogram,1,1,0,0,0,0.3,0.6,0.01,0,false\n"
            "histogram,2,2,1,0,0,0.7,0.85,0.01,0,false\n"
            "histogram,3,3,1,0,0,0.9,0.95,0.01,0,false\n"
        )

        with patch(
            "khisto.core.cli_adapter.tempfile.mkdtemp", return_value=str(tmp_path)
        ):
            with patch(
                "khisto.core.cli_adapter.tempfile.NamedTemporaryFile"
            ) as mock_temp:
                mock_file = MagicMock()
                mock_file.name = str(tmp_path / "input.txt")
                mock_file.__enter__ = MagicMock(return_value=mock_file)
                mock_file.__exit__ = MagicMock(return_value=False)
                mock_temp.return_value = mock_file

                # Request granularity 1 (second histogram)
                result = compute_histogram(sample_data, granularity=1)

                # Should only have rows from granularity 1
                granularities = set(result["granularity"].to_pylist())
                assert granularities == {1}
                assert len(result) == 2  # histogram.2.csv has 2 rows


class TestProcessHistogramFiles:
    """Tests for _process_histogram_files function."""

    def test_process_single_best_histogram(self, tmp_path):
        """Test processing a single best histogram file."""
        (tmp_path / "histogram.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n0.0,5.0,5.0,10,1.0,0.2\n"
        )

        result = _process_histogram_files(str(tmp_path), "histogram", only_best=True)

        assert len(result) == 1
        assert result["lower_bound"][0].as_py() == 0.0
        assert result["upper_bound"][0].as_py() == 5.0
        assert result["frequency"][0].as_py() == 10
        assert result["is_best"][0].as_py() is True
        assert result["granularity"][0].as_py() == 0

    def test_process_multiple_histograms_with_series(self, tmp_path):
        """Test processing multiple histograms with series data."""
        (tmp_path / "histogram.1.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n0.0,10.0,10.0,20,1.0,0.1\n"
        )
        (tmp_path / "histogram.2.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n"
            "0.0,5.0,5.0,10,0.5,0.1\n"
            "5.0,10.0,5.0,10,0.5,0.1\n"
        )
        (tmp_path / "histogram.series.csv").write_text(
            "FileName,Granularity,IntervalNumber,PeakIntervalNumber,SpikeIntervalNumber,"
            "EmptyIntervalNumber,Level,InformationRate,TruncationEpsilon,"
            "RemovedSingularityNumber,Raw\n"
            "histogram,1,1,0,0,0,0.3,0.5,0.01,0,false\n"
            "histogram,2,2,1,0,0,0.7,0.9,0.01,0,false\n"
        )

        result = _process_histogram_files(str(tmp_path), "histogram", only_best=False)

        # Should have 3 rows total (1 from granularity 0, 2 from granularity 1)
        assert len(result) == 3

        # Check granularities
        assert 0 in result["granularity"].to_pylist()
        assert 1 in result["granularity"].to_pylist()

        # Best should be granularity 1 (highest level = 0.7 from Granularity 2)
        best_rows = result.filter(result["is_best"])
        assert all(g == 1 for g in best_rows["granularity"].to_pylist())

        # Non-best rows should be granularity 0
        non_best_rows = result.filter(compute.invert(result["is_best"]))
        assert all(g == 0 for g in non_best_rows["granularity"].to_pylist())

    def test_column_renaming(self, tmp_path):
        """Test that columns are correctly renamed from PascalCase to snake_case."""
        (tmp_path / "histogram.csv").write_text(
            "LowerBound,UpperBound,Length,Frequency,Probability,Density\n1.0,2.0,1.0,5,0.5,0.5\n"
        )

        result = _process_histogram_files(str(tmp_path), "histogram", only_best=True)

        expected_columns = {
            "lower_bound",
            "upper_bound",
            "length",
            "frequency",
            "probability",
            "density",
            "granularity",
            "is_best",
        }
        assert set(result.column_names) == expected_columns

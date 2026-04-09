# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Core module for computing optimal histograms using the khisto CLI.

This module provides the interface to the khisto binary for computing
optimal histograms using the Khiops algorithm.
"""

from __future__ import annotations

import csv
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from khisto import KHISTO_BIN_DIR, logger


@dataclass
class HistogramResult:
    """Result of optimal histogram computation.

    Attributes
    ----------
    lower_bound : NDArray[np.float64]
        Lower bounds of each bin.
    upper_bound : NDArray[np.float64]
        Upper bounds of each bin.
    frequency : NDArray[np.int64]
        Count of values in each bin.
    probability : NDArray[np.float64]
        Probability of each bin (frequency / total).
    density : NDArray[np.float64]
        Density of each bin (probability / bin_width).
    is_best : bool
        Whether this histogram is the optimal one.
    granularity : int
        The granularity level of this histogram.
    """

    lower_bound: NDArray[np.float64]
    upper_bound: NDArray[np.float64]
    frequency: NDArray[np.int64]
    probability: NDArray[np.float64]
    density: NDArray[np.float64]
    is_best: bool = False
    granularity: int = 0

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        """Return bin edges array (n_bins + 1 values)."""
        return np.concatenate([self.lower_bound, [self.upper_bound[-1]]])

    @property
    def bin_widths(self) -> NDArray[np.float64]:
        """Return width of each bin."""
        return self.upper_bound - self.lower_bound

    @property
    def bin_centers(self) -> NDArray[np.float64]:
        """Return center of each bin."""
        return (self.lower_bound + self.upper_bound) / 2

    def __len__(self) -> int:
        """Return number of bins."""
        return len(self.lower_bound)


def _parse_file_type(
    file_path: pathlib.Path,
) -> Tuple[Literal["best_histogram", "histogram", "series"], Optional[int]]:
    """Determine the type of histogram file based on its name.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the histogram file.

    Returns
    -------
    tuple of (str, int or None)
        A tuple containing:
        - file_type : {'best_histogram', 'histogram', 'series'}
            Type of the histogram file.
        - histogram_id : int or None
            Histogram ID number for "histogram" type, None otherwise.

    Raises
    ------
    ValueError
        If the filename pattern is not recognized.
    """
    name = file_path.name

    if name == "histogram.csv":
        return "best_histogram", None
    if name == "histogram.series.csv":
        return "series", None

    # Check for histogram.(number).csv pattern
    stem = file_path.stem
    if stem.startswith("histogram.") and stem.split(".")[-1].isdigit():
        return "histogram", int(stem.split(".")[-1]) - 1

    raise ValueError(f"Unrecognized histogram file name: {name}")


def _read_csv_to_arrays(file_path: pathlib.Path) -> dict[str, np.ndarray]:
    """Read CSV file into dictionary of numpy arrays.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to CSV file.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping column names to numpy arrays.
    """
    with open(file_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {}

    result = {}
    columns = rows[0].keys()

    float_columns = {
        "Density",
        "InformationRate",
        "Length",
        "Level",
        "LowerBound",
        "Probability",
        "Raw",
        "TruncationEpsilon",
        "UpperBound",
    }
    int_columns = {
        "EmptyIntervalNumber",
        "Frequency",
        "Granularity",
        "IntervalNumber",
        "PeakIntervalNumber",
        "RemovedSingularityNumber",
        "SpikeIntervalNumber",
    }

    for col in columns:
        if col in float_columns:
            result[col] = np.array([float(row[col]) for row in rows], dtype=np.float64)
        elif col in int_columns:
            result[col] = np.array([int(row[col]) for row in rows], dtype=np.int64)
        else:
            # Keep as string for unknown columns
            result[col] = np.array([row[col] for row in rows], dtype=object)

    return result


def _build_histogram_result(
    data: dict[str, np.ndarray],
    *,
    is_best: bool,
    granularity: int,
) -> HistogramResult:
    """Build a histogram result object from parsed CSV data."""
    return HistogramResult(
        lower_bound=data["LowerBound"].astype(np.float64),
        upper_bound=data["UpperBound"].astype(np.float64),
        frequency=data["Frequency"].astype(np.int64),
        probability=data["Probability"].astype(np.float64),
        density=data["Density"].astype(np.float64),
        is_best=is_best,
        granularity=granularity,
    )


def _same_histogram(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> bool:
    """Return True when two parsed histogram CSV payloads describe the same bins."""
    comparable_columns = (
        "LowerBound",
        "UpperBound",
        "Frequency",
        "Probability",
        "Density",
    )
    return all(
        np.array_equal(left[column], right[column]) for column in comparable_columns
    )


def _process_histogram_files(
    temp_dir: str,
    base_name: str,
) -> list[HistogramResult]:
    """Process histogram files generated by khisto CLI.

    Parameters
    ----------
    temp_dir : str
        Directory containing histogram files.
    base_name : str
        Base name of histogram files (without extension).

    Returns
    -------
    list[HistogramResult]
        List of histogram results at all granularity levels,
        sorted from coarsest to finest.
    """
    best_histogram_data: Optional[dict[str, np.ndarray]] = None
    histogram_data: list[tuple[int, dict[str, np.ndarray]]] = []
    series_data: Optional[dict[str, np.ndarray]] = None

    # Read and process all histogram files
    for file in pathlib.Path(temp_dir).glob(f"{base_name}*"):
        ftype, hist_id = _parse_file_type(file)

        if ftype == "best_histogram":
            best_histogram_data = _read_csv_to_arrays(file)

        elif ftype == "histogram":
            data = _read_csv_to_arrays(file)
            assert hist_id is not None  # histogram type always has an id
            histogram_data.append((hist_id, data))

        elif ftype == "series":
            series_data = _read_csv_to_arrays(file)

    # Sort by granularity
    histogram_data.sort(key=lambda x: x[0])

    if not histogram_data:
        raise ValueError("No histogram data found")

    # Fallback: determine best granularity from the summary series when the
    # dedicated best histogram file does not match any numbered histogram file.
    best_granularity: Optional[int] = None
    if series_data is not None:
        max_level_idx = int(np.argmax(series_data["Level"]))
        best_granularity = int(series_data["Granularity"][max_level_idx]) - 1

    # Build all histogram results
    results = []
    for granularity, data in histogram_data:
        is_best = False
        if best_histogram_data is not None:
            is_best = _same_histogram(data, best_histogram_data)
        elif best_granularity is not None:
            is_best = granularity == best_granularity

        results.append(
            _build_histogram_result(data, is_best=is_best, granularity=granularity)
        )
    return results


def compute_histogram(
    x: np.ndarray,
) -> list[HistogramResult]:
    """Compute optimal histogram of an array using khisto CLI.

    Parameters
    ----------
    x : np.ndarray
        Array of numeric values.

    Returns
    -------
    list[HistogramResult]
        A list of HistogramResult, one per granularity level,
        sorted from coarsest to finest. Each result contains:
        - lower_bound : array of lower bin bounds
        - upper_bound : array of upper bin bounds
        - frequency : count in each bin
        - probability : probability of each bin
        - density : density of each bin
        - is_best : whether this is the optimal histogram
        - granularity : the granularity level

    Raises
    ------
    RuntimeError
        If khisto CLI execution fails.
    TypeError
        If input array is not numeric.
    ValueError
        If input array is empty after filtering.
    """
    # Convert to numpy array if needed
    x = np.asarray(x, dtype=np.float64)

    # Remove NaN values
    x = x[~np.isnan(x)]

    if len(x) == 0:
        raise ValueError("Input array is empty after filtering")

    # Create temporary input file with array values
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as temp_input:
        for val in x:
            temp_input.write(f"{val}\n")
        temp_input_path = temp_input.name

    # Create temporary directory for output files
    temp_dir = tempfile.mkdtemp(prefix="khisto_output_")
    output_file = pathlib.Path(temp_dir) / "histogram.csv"
    cmd = []

    try:
        # Build command arguments - always use exploratory mode
        cmd = [str(KHISTO_BIN_DIR), "-e", temp_input_path, str(output_file)]

        # Execute khisto CLI
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        result = _process_histogram_files(
            temp_dir,
            output_file.stem,
        )

        return result

    except subprocess.CalledProcessError as e:
        stdout = e.stdout.strip()
        stderr = e.stderr.strip()
        details = "\n".join(part for part in (stdout, stderr) if part)
        message = f"khisto failed with exit code {e.returncode} while running: {' '.join(cmd)}"
        if details:
            message = f"{message}\n{details}"
        logger.error(message)
        raise RuntimeError(message) from e
    finally:
        # Clean up temporary input file and directory
        pathlib.Path(temp_input_path).unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

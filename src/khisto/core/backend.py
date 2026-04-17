# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Backend module for computing optimal histograms using the khisto CLI.

This module provides the interface to the khisto binary for computing
optimal histograms using the Khiops algorithm.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Optional

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
) -> None:
    """Validate the exploratory JSON output file name.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to the JSON output file.

    Raises
    ------
    ValueError
        If the filename pattern is not recognized.
    """
    if file_path.suffix != ".json" or ".series" not in file_path.stem:
        raise ValueError(f"Unrecognized histogram file name: {file_path.name}")


def _json_float_array(payload: dict[str, Any], key: str) -> NDArray[np.float64]:
    """Read a JSON numeric list as a float64 array."""
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Missing or invalid histogram field: {key}")
    return np.asarray(values, dtype=np.float64)


def _json_int_array(payload: dict[str, Any], key: str) -> NDArray[np.int64]:
    """Read a JSON numeric list as an int64 array."""
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"Missing or invalid histogram field: {key}")
    return np.asarray(values, dtype=np.int64)


def _build_histogram_result(
    data: dict[str, Any],
    *,
    is_best: bool,
    granularity: int,
) -> HistogramResult:
    """Build a histogram result object from parsed JSON data."""
    lower_bound = _json_float_array(data, "lowerBounds")
    upper_bound = _json_float_array(data, "upperBounds")
    frequency = _json_int_array(data, "frequencies")
    probability = _json_float_array(data, "probabilities")
    density = _json_float_array(data, "densities")

    expected_size = len(lower_bound)
    if not all(
        len(array) == expected_size
        for array in (upper_bound, frequency, probability, density)
    ):
        raise ValueError("Inconsistent histogram payload lengths")

    return HistogramResult(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        frequency=frequency,
        probability=probability,
        density=density,
        is_best=is_best,
        granularity=granularity,
    )


def _same_histogram(
    left: HistogramResult,
    right: HistogramResult,
) -> bool:
    """Return True when two histogram results describe the same bins."""
    return all(
        np.array_equal(left_array, right_array)
        for left_array, right_array in (
            (left.lower_bound, right.lower_bound),
            (left.upper_bound, right.upper_bound),
            (left.frequency, right.frequency),
            (left.probability, right.probability),
            (left.density, right.density),
        )
    )


def _best_histogram_index_from_series(series_data: dict[str, Any]) -> Optional[int]:
    """Return the finest interpretable histogram index from the series metadata."""
    interpretable_histogram_number = series_data.get("interpretableHistogramNumber")
    if not isinstance(interpretable_histogram_number, int):
        return None
    if interpretable_histogram_number <= 0:
        return None
    return interpretable_histogram_number - 1


def _read_histogram_json(file_path: pathlib.Path) -> dict[str, Any]:
    """Read the khisto exploratory JSON output file."""
    with file_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)

    if not isinstance(payload, dict):
        raise ValueError("Invalid histogram JSON payload")
    return payload


def _process_histogram_files(
    temp_dir: str,
    base_name: str,
) -> list[HistogramResult]:
    """Process exploratory JSON generated by khisto CLI.

    Parameters
    ----------
    temp_dir : str
        Directory containing histogram files.
    base_name : str
        Base name of the histogram JSON file (without extension).

    Returns
    -------
    list[HistogramResult]
        List of histogram results at all granularity levels,
        sorted from coarsest to finest.
    """
    output_file = pathlib.Path(temp_dir) / f"{base_name}.json"
    _parse_file_type(output_file)

    payload = _read_histogram_json(output_file)
    best_histogram_payload = payload.get("bestHistogram")
    best_histogram = None
    if isinstance(best_histogram_payload, dict):
        best_histogram = _build_histogram_result(
            best_histogram_payload,
            is_best=False,
            granularity=0,
        )

    series_data = payload.get("histogramSeries")
    if not isinstance(series_data, dict):
        if best_histogram is None:
            raise ValueError("No histogram data found")
        best_histogram.is_best = True
        return [best_histogram]

    histogram_payloads = series_data.get("histograms")
    if not isinstance(histogram_payloads, list) or not histogram_payloads:
        if best_histogram is None:
            raise ValueError("No histogram data found")
        best_histogram.is_best = True
        return [best_histogram]

    granularities_payload = series_data.get("granularities")
    if isinstance(granularities_payload, list) and len(granularities_payload) == len(
        histogram_payloads
    ):
        granularities = [int(granularity) for granularity in granularities_payload]
    else:
        granularities = list(range(len(histogram_payloads)))

    best_index = _best_histogram_index_from_series(series_data)

    results = []
    for index, (granularity, histogram_payload) in enumerate(
        zip(granularities, histogram_payloads)
    ):
        if not isinstance(histogram_payload, dict):
            raise ValueError("Invalid histogram payload in histogramSeries")

        result = _build_histogram_result(
            histogram_payload,
            is_best=best_index == index,
            granularity=granularity,
        )

        if not result.is_best and best_index is None and best_histogram is not None:
            result.is_best = _same_histogram(result, best_histogram)

        results.append(result)

    if results:
        return results

    raise ValueError("No histogram data found")


def compute_histograms(
    x: np.ndarray,
) -> list[HistogramResult]:
    """Compute optimal histogram of an array using khisto CLI binary input.

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

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".bin", delete=False
    ) as temp_input:
        x.tofile(temp_input)
        temp_input_path = temp_input.name

    # Create temporary directory for output files
    temp_dir = tempfile.mkdtemp(prefix="khisto_output_")
    output_file = pathlib.Path(temp_dir) / "histogram.series.json"
    cmd = []

    try:
        # Build command arguments - always use exploratory mode
        cmd = [
            str(KHISTO_BIN_DIR),
            "-b",
            "-e",
            "-j",
            temp_input_path,
            str(output_file),
        ]

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

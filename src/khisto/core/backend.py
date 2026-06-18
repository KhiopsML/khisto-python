# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Backend module for computing optimal histograms using the khisto CLI."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from khisto import KHISTO_BIN_DIR, logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@dataclass
class _HistogramPayload:
    """Histogram bin data from khisto JSON."""

    lower_bounds: list[float] = field(default_factory=list)
    upper_bounds: list[float] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    frequencies: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    densities: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _HistogramPayload:
        return cls(
            **{
                ck: v
                for k, v in data.items()
                if (ck := camel_to_snake(k)) in cls.__dataclass_fields__
            }
        )


@dataclass
class _SeriesPayload:
    """Histogram series from khisto JSON."""

    histogram_number: int = 0
    interpretable_histogram_number: int = 0
    truncation_epsilon: float = 0.0
    removed_singular_interval_number: int = 0
    granularities: list[int] = field(default_factory=list)
    interval_numbers: list[int] = field(default_factory=list)
    peak_interval_numbers: list[int] = field(default_factory=list)
    spike_interval_numbers: list[int] = field(default_factory=list)
    empty_interval_numbers: list[int] = field(default_factory=list)
    levels: list[float] = field(default_factory=list)
    information_rates: list[float] = field(default_factory=list)
    histograms: list[_HistogramPayload] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _SeriesPayload:
        kwargs = {
            ck: v
            for k, v in data.items()
            if (ck := camel_to_snake(k)) in cls.__dataclass_fields__
        }
        if "histograms" in kwargs:
            kwargs["histograms"] = [
                _HistogramPayload.from_dict(h) for h in kwargs["histograms"]
            ]
        return cls(**kwargs)


@dataclass
class _KhistoOutput:
    """Root khisto JSON output."""

    tool: str = ""
    version: str = ""
    best_histogram: _HistogramPayload = field(default_factory=_HistogramPayload)
    histogram_series: _SeriesPayload = field(default_factory=_SeriesPayload)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _KhistoOutput:
        if "bestHistogram" not in data or "histogramSeries" not in data:
            raise ValueError("Missing required fields: bestHistogram, histogramSeries")
        return cls(
            tool=data.get("tool", ""),
            version=data.get("version", ""),
            best_histogram=_HistogramPayload.from_dict(data["bestHistogram"]),
            histogram_series=_SeriesPayload.from_dict(data["histogramSeries"]),
        )


@dataclass
class HistogramResult:
    """Result of optimal histogram computation.

    Attributes
    ----------
    lower_bounds : NDArray[np.float64]
        Lower bounds of each bin.
    upper_bounds : NDArray[np.float64]
        Upper bounds of each bin.
    frequencies : NDArray[np.int64]
        Count of values in each bin.
    probabilities : NDArray[np.float64]
        Probability of each bin (frequency / total).
    densities : NDArray[np.float64]
        Density of each bin (probability / bin_width).
    is_best : bool
        Whether this histogram is the optimal one.
    granularity : int
        The granularity level of this histogram.
    level : float
        The information level of this histogram.
    information_rate : float
        The information rate of this histogram (percentage of the finest level).
    peak_interval_number : int
        Number of peak intervals in this histogram.
    spike_interval_number : int
        Number of spike intervals in this histogram.
    empty_interval_number : int
        Number of empty intervals in this histogram.
    """

    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    frequencies: NDArray[np.int64]
    probabilities: NDArray[np.float64]
    densities: NDArray[np.float64]
    is_best: bool = False
    granularity: int = 0
    level: float = 0.0
    information_rate: float = 0.0
    peak_interval_number: int = 0
    spike_interval_number: int = 0
    empty_interval_number: int = 0

    def __post_init__(self) -> None:
        """Validate the histogram data."""

        # Verify all arrays have the same length
        n_bins = {
            len(self.lower_bounds),
            len(self.upper_bounds),
            len(self.frequencies),
            len(self.probabilities),
            len(self.densities),
        }
        if len(n_bins) != 1:
            raise ValueError("all array sizes must be equal")
        elif len(self.lower_bounds) < 1:
            raise ValueError("all arrays must have at least one element")

        if not np.all(self.lower_bounds <= self.upper_bounds):
            raise ValueError("lower_bounds must be less than upper_bounds")
        if not np.all(self.frequencies >= 0):
            raise ValueError("frequencies must be non-negative")
        if not np.all(self.probabilities >= 0):
            raise ValueError("probabilities must be non-negative")
        if not np.all(self.densities >= 0):
            raise ValueError("densities must be non-negative")

        # Verify lower_bounds are equal to upper_bounds for adjacent bins
        if not np.all(self.lower_bounds[1:] == self.upper_bounds[:-1]):
            raise ValueError(
                "lower_bounds must be equal to upper_bounds for adjacent bins"
            )

    @property
    def bin_edges(self) -> NDArray[np.float64]:
        """Return bin edges array (n_bins + 1 values).
        lower_bounds and upper_bounds are equal for adjacent bins."""
        return np.concatenate([self.lower_bounds, [self.upper_bounds[-1]]])

    @property
    def bin_widths(self) -> NDArray[np.float64]:
        """Return width of each bin."""
        return self.upper_bounds - self.lower_bounds

    @property
    def bin_centers(self) -> NDArray[np.float64]:
        """Return center of each bin."""
        return (self.lower_bounds + self.upper_bounds) / 2

    def __len__(self) -> int:
        """Return number of bins."""
        return len(self.lower_bounds)


def _format_runtime_error(
    summary: str, cmd: list[str], details: str | None = None
) -> str:
    """Build a consistent runtime error message for khisto failures."""
    message = f"{summary} while running: {' '.join(cmd)}"
    if details:
        message = f"{message}\n{details}"
    return message


def _process_histogram_file(file_path: Path) -> list[HistogramResult]:
    """Process exploratory JSON generated by khisto CLI."""
    with open(file_path, "r", encoding="utf-8") as file:
        khisto_output: _KhistoOutput = _KhistoOutput.from_dict(json.load(file))

    histogram_series = khisto_output.histogram_series
    best_idx = histogram_series.interpretable_histogram_number - 1

    return [
        HistogramResult(
            lower_bounds=np.asarray(h.lower_bounds, dtype=np.float64),
            upper_bounds=np.asarray(h.upper_bounds, dtype=np.float64),
            frequencies=np.asarray(h.frequencies, dtype=np.int64),
            probabilities=np.asarray(h.probabilities, dtype=np.float64),
            densities=np.asarray(h.densities, dtype=np.float64),
            is_best=(i == best_idx),
            granularity=histogram_series.granularities[i],
            level=histogram_series.levels[i],
            information_rate=histogram_series.information_rates[i],
            peak_interval_number=histogram_series.peak_interval_numbers[i],
            spike_interval_number=histogram_series.spike_interval_numbers[i],
            empty_interval_number=histogram_series.empty_interval_numbers[i],
        )
        for i, h in enumerate(histogram_series.histograms)
    ]


def compute_histograms(x: np.ndarray) -> list[HistogramResult]:
    """Compute optimal histogram of an array using khisto CLI binary input.

    Parameters
    ----------
    x : np.ndarray
        Array of numeric values.

    Returns
    -------
    list[HistogramResult]
        A list of HistogramResult, one per granularity level,
        sorted from coarsest to finest.

    Raises
    ------
    RuntimeError
        If khisto CLI execution fails or returns invalid output.
    ValueError
        If input array is empty after filtering.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        raise ValueError("Input array is empty after filtering")

    # Use delete=False so the files are closed before the subprocess reads them.
    # On Windows, NamedTemporaryFile keeps an exclusive lock while open.
    temp_input_file = tempfile.NamedTemporaryFile(
        mode="wb", suffix=".bin", delete=False
    )
    temp_output_file = tempfile.NamedTemporaryFile(
        mode="r", suffix=".json", delete=False
    )
    try:
        x.tofile(temp_input_file)
        temp_input_file.close()
        temp_output_file.close()

        cmd = [
            str(KHISTO_BIN_DIR),
            "-b",
            "-e",
            "-j",
            temp_input_file.name,
            temp_output_file.name,
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            stdout = e.stdout.strip()
            stderr = e.stderr.strip()
            details = "\n".join(part for part in (stdout, stderr) if part)
            message = _format_runtime_error(
                f"khisto failed with exit code {e.returncode}",
                cmd,
                details or None,
            )
            logger.error(message)
            raise RuntimeError(message) from e
        except OSError as e:
            message = _format_runtime_error(
                "khisto could not be started",
                cmd,
                str(e),
            )
            logger.error(message)
            raise RuntimeError(message) from e

        try:
            return _process_histogram_file(Path(temp_output_file.name))
        except json.JSONDecodeError as e:
            message = _format_runtime_error(
                "khisto produced invalid JSON output",
                cmd,
                str(e),
            )
            logger.error(message)
            raise RuntimeError(message) from e
        except (AttributeError, IndexError, TypeError, ValueError) as e:
            message = _format_runtime_error(
                "khisto produced an invalid histogram payload",
                cmd,
                str(e),
            )
            logger.error(message)
            raise RuntimeError(message) from e
    finally:
        os.unlink(temp_input_file.name)
        os.unlink(temp_output_file.name)

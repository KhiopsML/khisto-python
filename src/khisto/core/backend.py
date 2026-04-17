# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Backend module for computing optimal histograms using the khisto CLI."""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from khisto import KHISTO_BIN_DIR, logger


@dataclass
class _HistogramPayload:
    """Histogram bin data from khisto JSON."""

    lowerBounds: list[float] = field(default_factory=list)
    upperBounds: list[float] = field(default_factory=list)
    lengths: list[float] = field(default_factory=list)
    frequencies: list[int] = field(default_factory=list)
    probabilities: list[float] = field(default_factory=list)
    densities: list[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _HistogramPayload:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class _SeriesPayload:
    """Histogram series from khisto JSON."""

    histogramNumber: int = 0
    interpretableHistogramNumber: int = 0
    truncationEpsilon: float = 0.0
    removedSingularIntervalNumber: int = 0
    granularities: list[int] = field(default_factory=list)
    intervalNumbers: list[int] = field(default_factory=list)
    peakIntervalNumbers: list[int] = field(default_factory=list)
    spikeIntervalNumbers: list[int] = field(default_factory=list)
    emptyIntervalNumbers: list[int] = field(default_factory=list)
    levels: list[float] = field(default_factory=list)
    informationRates: list[float] = field(default_factory=list)
    histograms: list[_HistogramPayload] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _SeriesPayload:
        kwargs = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
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
    bestHistogram: Optional[_HistogramPayload] = None
    histogramSeries: Optional[_SeriesPayload] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _KhistoOutput:
        best = data.get("bestHistogram")
        series = data.get("histogramSeries")
        return cls(
            tool=data.get("tool", ""),
            version=data.get("version", ""),
            bestHistogram=_HistogramPayload.from_dict(best) if best else None,
            histogramSeries=_SeriesPayload.from_dict(series) if series else None,
        )


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

    lower_bound: NDArray[np.float64]
    upper_bound: NDArray[np.float64]
    frequency: NDArray[np.int64]
    probability: NDArray[np.float64]
    density: NDArray[np.float64]
    is_best: bool = False
    granularity: int = 0
    level: float = 0.0
    information_rate: float = 0.0
    peak_interval_number: int = 0
    spike_interval_number: int = 0
    empty_interval_number: int = 0

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


def _to_result(h: _HistogramPayload, **kwargs: Any) -> HistogramResult:
    """Convert a JSON histogram payload to a HistogramResult."""
    return HistogramResult(
        lower_bound=np.asarray(h.lowerBounds, dtype=np.float64),
        upper_bound=np.asarray(h.upperBounds, dtype=np.float64),
        frequency=np.asarray(h.frequencies, dtype=np.int64),
        probability=np.asarray(h.probabilities, dtype=np.float64),
        density=np.asarray(h.densities, dtype=np.float64),
        **kwargs,
    )


def _format_runtime_error(
    summary: str, cmd: list[str], details: str | None = None
) -> str:
    """Build a consistent runtime error message for khisto failures."""
    message = f"{summary} while running: {' '.join(cmd)}"
    if details:
        message = f"{message}\n{details}"
    return message


def _process_histogram_file(
    temp_output_file: tempfile._TemporaryFileWrapper[str],
) -> list[HistogramResult]:
    """Process exploratory JSON generated by khisto CLI."""
    temp_output_file.seek(0)
    khisto_output: _KhistoOutput = _KhistoOutput.from_dict(json.load(temp_output_file))

    histogram_series = khisto_output.histogramSeries
    best_idx = histogram_series.interpretableHistogramNumber - 1

    return [
        _to_result(
            h,
            is_best=(i == best_idx),
            granularity=histogram_series.granularities[i],
            level=histogram_series.levels[i],
            information_rate=histogram_series.informationRates[i],
            peak_interval_number=histogram_series.peakIntervalNumbers[i],
            spike_interval_number=histogram_series.spikeIntervalNumbers[i],
            empty_interval_number=histogram_series.emptyIntervalNumbers[i],
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

    with tempfile.NamedTemporaryFile(mode="wb", suffix=".bin") as temp_input_file:
        x.tofile(temp_input_file)
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json") as temp_output_file:
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
                return _process_histogram_file(temp_output_file)
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

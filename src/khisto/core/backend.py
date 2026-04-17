# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Backend module for computing optimal histograms using the khisto CLI."""

from __future__ import annotations

import json
import pathlib
import shutil
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


def _parse_file_type(file_path: pathlib.Path) -> None:
    """Validate the exploratory JSON output file name."""
    if file_path.suffix != ".json" or ".series" not in file_path.stem:
        raise ValueError(f"Unrecognized histogram file name: {file_path.name}")


def _process_histogram_files(temp_dir: str, base_name: str) -> list[HistogramResult]:
    """Process exploratory JSON generated by khisto CLI."""
    output_file = pathlib.Path(temp_dir) / f"{base_name}.json"
    _parse_file_type(output_file)

    with output_file.open("r", encoding="utf-8") as f:
        khisto_output: _KhistoOutput = _KhistoOutput.from_dict(json.load(f))

    histogram_series = khisto_output.histogramSeries
    n = len(histogram_series.histograms)
    best_idx = histogram_series.interpretableHistogramNumber - 1
    granularities = (
        histogram_series.granularities
        if len(histogram_series.granularities) == n
        else list(range(n))
    )

    def _get(lst: list, i: int):
        if i < len(lst):
            return lst[i]
        else:
            raise ValueError(
                f"Expected at least {i + 1} values in list, got {len(lst)}"
            )

    return [
        _to_result(
            h,
            is_best=(i == best_idx),
            granularity=granularities[i],
            level=_get(histogram_series.levels, i),
            information_rate=_get(histogram_series.informationRates, i),
            peak_interval_number=_get(histogram_series.peakIntervalNumbers, i),
            spike_interval_number=_get(histogram_series.spikeIntervalNumbers, i),
            empty_interval_number=_get(histogram_series.emptyIntervalNumbers, i),
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
        If khisto CLI execution fails.
    ValueError
        If input array is empty after filtering.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        raise ValueError("Input array is empty after filtering")

    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".bin", delete=False
    ) as temp_input:
        x.tofile(temp_input)
        temp_input_path = temp_input.name

    temp_dir = tempfile.mkdtemp(prefix="khisto_output_")
    output_file = pathlib.Path(temp_dir) / "histogram.series.json"
    cmd: list[str] = []

    try:
        cmd = [
            str(KHISTO_BIN_DIR),
            "-b",
            "-e",
            "-j",
            temp_input_path,
            str(output_file),
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return _process_histogram_files(temp_dir, output_file.stem)

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
        pathlib.Path(temp_input_path).unlink(missing_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)

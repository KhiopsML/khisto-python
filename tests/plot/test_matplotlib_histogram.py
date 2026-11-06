# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Tests for matplotlib histogram plotting."""

import builtins
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import assert_type

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.units as munits
from matplotlib.colors import to_rgba
from matplotlib.container import BarContainer
from matplotlib.patches import Polygon

from khisto import histogram
from khisto.matplotlib import hist


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure, including when a test fails."""
    yield
    plt.close("all")


def test_missing_matplotlib_reports_install_extra(monkeypatch):
    """Test that plotting without Matplotlib explains how to install it."""
    original_import = builtins.__import__
    missing_matplotlib = ModuleNotFoundError(
        "No module named 'matplotlib'", name="matplotlib"
    )

    def import_without_matplotlib(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise missing_matplotlib
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)

    with pytest.raises(
        ImportError, match=r"pip install 'khisto\[matplotlib\]'"
    ) as excinfo:
        hist([1.0, 2.0, 3.0])
    assert excinfo.value.__cause__ is missing_matplotlib


def test_matplotlib_dependency_import_error_is_chained(monkeypatch):
    """Test that broken Matplotlib dependencies remain available as the cause."""
    original_import = builtins.__import__
    expected_error = ModuleNotFoundError(
        "No module named 'matplotlib_dependency'", name="matplotlib_dependency"
    )

    def import_without_dependency(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise expected_error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_dependency)

    with pytest.raises(ImportError, match=r"Plotting requires Matplotlib") as excinfo:
        hist([1.0, 2.0, 3.0])

    assert excinfo.value.__cause__ is expected_error


class TestHistBasic:
    """Test basic hist functionality."""

    @pytest.fixture
    def normal_data(self):
        """Normal distribution data"""
        np.random.seed(42)
        return np.random.normal(0, 1, 1000)

    def test_simple_array(self, normal_data):
        """Test hist with simple numpy array."""
        _fig, ax = plt.subplots()
        n, bins, patches = hist(normal_data, ax=ax)

        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert patches is not None
        assert len(n) > 0
        assert len(bins) == len(n) + 1

    def test_without_ax(self, normal_data):
        """Test hist without explicit ax parameter."""
        n, bins, _ = hist(normal_data)

        assert isinstance(n, np.ndarray)
        assert isinstance(bins, np.ndarray)
        assert len(n) > 0

    def test_density_histogram(self, normal_data):
        """Test density histogram."""
        _fig, ax = plt.subplots()
        n, bins, _ = hist(normal_data, density=True, ax=ax)

        # Density should integrate to 1
        bin_widths = np.diff(bins)
        total = np.sum(n * bin_widths)
        assert np.isclose(total, 1.0, rtol=1e-5)

    def test_frequency_histogram(self, normal_data):
        """Test frequency histogram is the default behavior."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, ax=ax, density=False)

        # Frequencies should sum to total count
        assert np.sum(n) == len(normal_data)

    @pytest.mark.parametrize("density", [False, True])
    def test_histogram_matches_khiops_at_internal_edges(self, density):
        """Test that observations on internal edges keep their Khiops bins."""
        data = np.repeat([1, 2, 3, 4, 5, 6], [458, 82, 43, 11, 3, 2])
        expected, expected_bins = histogram(data, max_bins=100, density=density)
        _fig, ax = plt.subplots()

        values, bins, _ = hist(data, max_bins=100, density=density, ax=ax)

        np.testing.assert_array_equal(bins, expected_bins)
        np.testing.assert_allclose(values, expected)

    def test_histogram_matches_khiops_for_large_values(self):
        """Test that plotting reuses counts when Khiops adjusts extreme edges."""
        data = np.array([1e300, np.nextafter(1e300, np.inf), 2e300])
        expected, expected_bins = histogram(data, density=False)
        _fig, ax = plt.subplots()

        values, bins, _ = hist(data, density=False, ax=ax)

        np.testing.assert_array_equal(bins, expected_bins)
        np.testing.assert_array_equal(values, expected)

    def test_horizontal_orientation(self, normal_data):
        """Test horizontal histogram."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, orientation="horizontal", ax=ax)

        assert isinstance(n, np.ndarray)
        assert len(n) > 0

    def test_with_max_bins(self, normal_data):
        """Test histogram with max_bins parameter."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, max_bins=5, ax=ax)

        assert len(n) <= 5

    def test_with_range(self, normal_data):
        """Test histogram with range parameter."""
        _fig, ax = plt.subplots()
        _, bins, _ = hist(normal_data, range=(-1, 1), ax=ax)

        assert bins[0] >= -1
        assert bins[-1] <= 1

    def test_log_scale(self, normal_data):
        """Test histogram with log scale."""
        _fig, ax = plt.subplots()
        hist(normal_data, log=True, ax=ax)

        assert ax.get_yscale() == "log"

    def test_color_parameter(self, normal_data):
        """Test histogram with color parameter."""
        _fig, ax = plt.subplots()
        _, _, patches = hist(normal_data, color="red", ax=ax)

        assert patches is not None

    @pytest.mark.parametrize(
        "alpha, expected_edge_alpha",
        [(None, 0.5), (0.5, 0.5), (0.0, 0.0), (1.0, 1.0)],
    )
    def test_bar_edges_match_face_rgb_with_default_opacity(
        self, normal_data, alpha, expected_edge_alpha
    ):
        """Test default edge opacity and explicit histogram alpha overrides."""
        _fig, ax = plt.subplots()
        _, _, patches = hist(normal_data, color="tab:blue", alpha=alpha, ax=ax)

        for patch in patches.patches:
            edgecolor = to_rgba(patch.get_edgecolor())
            facecolor = to_rgba(patch.get_facecolor())
            assert edgecolor[:3] == facecolor[:3]
            assert edgecolor[3] == pytest.approx(expected_edge_alpha)
            assert facecolor[3] == pytest.approx(1.0 if alpha is None else alpha)
            assert patch.get_alpha() == alpha

    @pytest.mark.parametrize("edgecolor_keyword", ["edgecolor", "ec"])
    @pytest.mark.parametrize(
        "edgecolor, expected_rgba",
        [
            ("red", (1.0, 0.0, 0.0, 1.0)),
            ("none", (0.0, 0.0, 0.0, 0.0)),
            ((0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4)),
        ],
    )
    def test_explicit_bar_edge_style_is_preserved(
        self, normal_data, edgecolor_keyword, edgecolor, expected_rgba
    ):
        """Test that explicit edge styling aliases override the khisto default."""
        _fig, ax = plt.subplots()
        _, _, patches = hist(
            normal_data,
            linewidth=2.0,
            ax=ax,
            **{edgecolor_keyword: edgecolor},
        )

        for patch in patches.patches:
            # Matplotlib returns edge colors as RGBA, including their opacity.
            assert patch.get_edgecolor() == expected_rgba
            assert patch.get_linewidth() == 2.0

    def test_step_histtype(self, normal_data):
        """Test histogram with step histtype."""
        _fig, ax = plt.subplots()
        n, _, patches = hist(normal_data, histtype="step", ax=ax)

        assert isinstance(n, np.ndarray)
        # assert_type checks static inference; isinstance checks runtime values.
        assert_type(patches, list[Polygon])
        assert all(isinstance(patch, Polygon) for patch in patches)

    def test_stepfilled_histtype(self, normal_data):
        """Test histogram with stepfilled histtype."""
        _fig, ax = plt.subplots()
        n, _, patches = hist(normal_data, histtype="stepfilled", ax=ax)

        assert isinstance(n, np.ndarray)
        # assert_type checks static inference; isinstance checks runtime values.
        assert_type(patches, list[Polygon])
        assert all(isinstance(patch, Polygon) for patch in patches)

    def test_cumulative_density_histogram(self, normal_data):
        """Test cumulative density histogram."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, density=True, cumulative=True, ax=ax)

        assert np.isclose(n[-1], 1.0, rtol=1e-5)

    def test_cumulative_hist_matches_array_api(self, normal_data):
        """Test that the plotting wrapper matches cumulative histogram values."""
        _fig, ax = plt.subplots()
        n, bins, _ = hist(normal_data, density=True, cumulative=True, ax=ax)
        density, expected_bins = histogram(normal_data, density=True)
        expected = np.cumsum(density * np.diff(expected_bins))

        np.testing.assert_array_equal(bins, expected_bins)
        np.testing.assert_allclose(n, expected)

    def test_reverse_cumulative_frequency_histogram(self, normal_data):
        """Test reverse cumulative frequency histogram."""
        _fig, ax = plt.subplots()
        n, _, _ = hist(normal_data, density=False, cumulative=-1, ax=ax)

        assert np.isclose(n[0], len(normal_data))

    @pytest.mark.parametrize(
        "name, value", [("bins", 10), ("stacked", False), ("weights", [])]
    )
    def test_unsupported_parameter(self, normal_data, name, value):
        """Test that unsupported parameters raise clear error messages."""
        _fig, ax = plt.subplots()

        with pytest.raises(TypeError, match=rf"{name} is not supported"):
            hist(normal_data, ax=ax, **{name: value})

    def test_unsupported_barstacked_histtype(self, normal_data):
        """Test that barstacked raises a clear error message."""
        _fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="barstacked.*not supported"):
            hist(normal_data, histtype="barstacked", ax=ax)


class TestHistUnits:
    """Test unit conversion before computing adaptive bins."""

    @pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
    @pytest.mark.parametrize("density", [False, True])
    @pytest.mark.parametrize(
        "kind", ["date", "datetime", "datetime64[D]", "datetime64[ns]"]
    )
    def test_dates_use_matplotlib_coordinates_and_formatting(
        self, orientation, density, kind
    ):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        data = [start + timedelta(days=i) for i in range(60)]
        if kind == "date":
            data = [value.date() for value in data]
        elif kind.startswith("datetime64"):
            data = np.asarray([value.date() for value in data], dtype=kind)
        expected, expected_bins = histogram(mdates.date2num(data), density=density)
        _, ax = plt.subplots()

        values, bins, patches = hist(
            data, ax=ax, density=density, orientation=orientation
        )

        assert_type(patches, BarContainer)
        axis = ax.yaxis if orientation == "horizontal" else ax.xaxis
        np.testing.assert_allclose(values, expected)
        np.testing.assert_array_equal(bins, expected_bins)
        assert isinstance(axis.get_major_locator(), mdates.AutoDateLocator)
        assert isinstance(axis.get_major_formatter(), mdates.AutoDateFormatter)
        ax.figure.canvas.draw()

    @pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
    @pytest.mark.parametrize("numeric_range", [False, True])
    def test_date_range_uses_the_data_axis_units(self, orientation, numeric_range):
        data = np.arange("2026-01-01", "2026-03-01", dtype="datetime64[D]")
        bounds = (date(2026, 1, 10), date(2026, 2, 10))
        converted_bounds = tuple(mdates.date2num(bounds))
        expected, expected_bins = histogram(
            mdates.date2num(data), range=converted_bounds, density=False
        )
        _, ax = plt.subplots()

        values, bins, _ = hist(
            data,
            range=converted_bounds if numeric_range else bounds,
            density=False,
            orientation=orientation,
            ax=ax,
        )

        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(bins, expected_bins)

    def test_decimal_matches_float_input(self):
        data = [Decimal(i) / 10 for i in range(100)]
        expected, expected_bins = histogram(np.asarray(data, dtype=float))

        values, bins, patches = hist(data)

        assert_type(patches, BarContainer)
        np.testing.assert_allclose(values, expected)
        np.testing.assert_array_equal(bins, expected_bins)

    @pytest.fixture
    def unit_value(self, monkeypatch):
        @dataclass
        class UnitValue:
            magnitude: float

        class Converter:
            @staticmethod
            def default_units(values, axis):
                return "m"

            @staticmethod
            def axisinfo(unit, axis):
                return munits.AxisInfo(label=f"Distance ({unit})")

            @staticmethod
            def convert(values, unit, axis):
                scale = 100 if unit == "cm" else 1
                return np.asarray([value.magnitude for value in values]) * scale

        monkeypatch.setitem(munits.registry, UnitValue, Converter())
        return UnitValue

    @pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
    @pytest.mark.parametrize("unit", ["m", "cm"])
    def test_registered_units_and_overrides(self, unit_value, orientation, unit):
        data = [unit_value(i) for i in range(100)]
        scale = 100 if unit == "cm" else 1
        expected, expected_bins = histogram(
            np.arange(100) * scale, range=(10 * scale, 80 * scale), density=False
        )
        _, ax = plt.subplots()
        unit_override = unit if unit == "cm" else None

        values, bins, patches = hist(
            data,
            range=(unit_value(10), unit_value(80)),
            density=False,
            orientation=orientation,
            ax=ax,
            xunits=unit_override if orientation == "vertical" else None,
            yunits=unit_override if orientation == "horizontal" else None,
        )

        assert_type(patches, BarContainer)
        axis = ax.yaxis if orientation == "horizontal" else ax.xaxis
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_array_equal(bins, expected_bins)
        assert axis.get_units() == unit
        assert axis.get_label_text() == f"Distance ({unit})"
        ax.figure.canvas.draw()


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
        # assert_type checks static inference; isinstance checks runtime values.
        assert_type(patches, BarContainer)
        assert isinstance(patches, BarContainer)

    def test_bins_edges_count(self, data):
        """Test that bins has n+1 edges."""
        n, bins, _ = hist(data)
        assert len(bins) == len(n) + 1

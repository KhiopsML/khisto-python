"""Tests for matplotlib cumulative distribution plotting."""

from __future__ import annotations

import numpy as np
import pytest

from khisto.matplotlib import ecdf

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt


class TestCumulativeBasic:
    """Test basic cumulative distribution functionality."""

    def test_simple_array(self):
        """Test cumulative plot with simple numpy array."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax)

        assert line is not None
        # line is a Line2D when hue is None
        if isinstance(line, list):
            line = line[0]
        assert hasattr(line, "get_xdata")
        assert hasattr(line, "get_ydata")
        plt.close(fig)

    def test_without_ax(self):
        """Test cumulative plot without explicit ax parameter."""
        data = np.random.normal(0, 1, 1000)
        line = ecdf(x=data)

        assert line is not None
        if isinstance(line, list):
            line = line[0]
        assert hasattr(line, "get_xdata")
        plt.close()

    def test_with_title(self):
        """Test cumulative plot with title."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        ecdf(x=data, ax=ax, title="Test Cumulative")

        assert ax.get_title() == "Test Cumulative"
        plt.close(fig)

    def test_horizontal_orientation(self):
        """Test horizontal cumulative plot."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, orientation="horizontal")

        assert line is not None
        if isinstance(line, list):
            line = line[0]
        assert hasattr(line, "get_xdata")
        plt.close(fig)

    def test_cumulative_values_range(self):
        """Test that cumulative values go from 0 to 1."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax)

        assert line is not None
        if isinstance(line, list):
            line = line[0]

        y_data = line.get_ydata()  # type: ignore[union-attr]
        assert y_data[0] == 0.0  # type: ignore[index]  # First point should be 0
        assert y_data[-1] == 1.0  # type: ignore[index]  # Last point should be 1
        assert all(0 <= y <= 1 for y in y_data)  # type: ignore[attr-defined]  # All values between 0 and 1
        plt.close(fig)


class TestCumulativeWithDataFrame:
    """Test cumulative plot with DataFrame input."""

    def test_pandas_dataframe(self):
        """Test cumulative plot with pandas DataFrame."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B"], 1000),
            }
        )

        fig, ax = plt.subplots()
        lines = ecdf(data=df, x="value", ax=ax)

        assert lines is not None
        plt.close(fig)

    def test_pandas_with_hue(self):
        """Test cumulative plot with pandas DataFrame and hue."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B"], 1000),
            }
        )

        fig, ax = plt.subplots()
        lines = ecdf(data=df, x="value", hue="category", ax=ax)
        assert lines is not None
        assert isinstance(lines, list)
        assert len(lines) == 2
        plt.close(fig)

    def test_polars_dataframe(self):
        """Test cumulative plot with polars DataFrame."""
        pl = pytest.importorskip("polars")

        df = pl.DataFrame(
            {
                "value": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B"], 1000),
            }
        )

        fig, ax = plt.subplots()
        lines = ecdf(data=df, x="value", ax=ax)

        assert lines is not None
        plt.close(fig)


class TestCumulativeStyling:
    """Test cumulative plot styling options."""

    def test_custom_color(self):
        """Test cumulative plot with custom color."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, color="red")
        assert line is not None
        if isinstance(line, list):
            assert len(line) > 0
            line = line[0]
        # Check that line has the specified color
        assert line.get_color() == "red"
        plt.close(fig)

    def test_alpha(self):
        """Test cumulative plot with alpha transparency."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, alpha=0.5)

        assert line is not None
        if isinstance(line, list):
            line = line[0]
        # Check that line has the specified alpha
        assert line.get_alpha() == 0.5
        plt.close(fig)

    def test_linewidth(self):
        """Test cumulative plot with custom line width."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, linewidth=2.0)

        assert line is not None
        if isinstance(line, list):
            line = line[0]
        # Check that line has the specified linewidth
        assert line.get_linewidth() == 2.0
        plt.close(fig)

    def test_linestyle(self):
        """Test cumulative plot with custom line style."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, linestyle="--")

        assert line is not None
        if isinstance(line, list):
            line = line[0]
        # Check that line has the specified linestyle
        assert line.get_linestyle() == "--"
        plt.close(fig)

    def test_marker(self):
        """Test cumulative plot with markers."""
        data = np.random.normal(0, 1, 100)  # Smaller dataset for markers
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, marker="o", markevery=10)
        assert line is not None
        if isinstance(line, list):
            assert len(line) > 0
            line = line[0]
        # Check that line has the specified marker
        assert line.get_marker() == "o"
        plt.close(fig)


class TestCumulativeLabels:
    """Test cumulative plot axis labels."""

    def test_custom_xlabel(self):
        """Test cumulative plot with custom x label."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        ecdf(x=data, ax=ax, xlabel="Custom X")

        assert ax.get_xlabel() == "Custom X"
        plt.close(fig)

    def test_custom_ylabel(self):
        """Test cumulative plot with custom y label."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        ecdf(x=data, ax=ax, ylabel="Custom Y")

        assert ax.get_ylabel() == "Custom Y"
        plt.close(fig)

    def test_default_labels_vertical(self):
        """Test default labels for vertical cumulative plot."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        ecdf(x=data, ax=ax)

        assert ax.get_xlabel() == "_value"
        assert ax.get_ylabel() == "Cumulative Probability"
        plt.close(fig)

    def test_default_labels_horizontal(self):
        """Test default labels for horizontal cumulative plot."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        ecdf(x=data, ax=ax, orientation="horizontal")

        assert ax.get_ylabel() == "_value"
        assert ax.get_xlabel() == "Cumulative Probability"
        plt.close(fig)


class TestCumulativeGranularity:
    """Test cumulative plot granularity options."""

    def test_best_granularity(self):
        """Test cumulative plot with best granularity."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, granularity="best")

        assert line is not None
        if isinstance(line, list):
            line = line[0]
        assert hasattr(line, "get_xdata")
        plt.close(fig)

    def test_specific_granularity(self):
        """Test cumulative plot with specific granularity level."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        line = ecdf(x=data, ax=ax, granularity=0)

        assert line is not None
        if isinstance(line, list):
            line = line[0]
        assert hasattr(line, "get_xdata")
        plt.close(fig)


class TestCumulativeErrors:
    """Test error handling."""

    def test_no_data_error(self):
        """Test that error is raised when no data provided."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Either 'data' or 'x' must be provided"):
            ecdf(ax=ax)
        plt.close(fig)

    def test_string_x_without_data_error(self):
        """Test that error is raised when x is string but no data."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="x must be an array"):
            ecdf(x="column_name", ax=ax)
        plt.close(fig)

    def test_no_x_with_data_error(self):
        """Test that error is raised when data provided but no x."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"value": [1, 2, 3]})

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="x column name must be specified"):
            ecdf(data=df, ax=ax)
        plt.close(fig)


class TestCumulativeAxisRanges:
    """Test that cumulative probability axis has correct range."""

    def test_vertical_y_axis_range(self):
        """Test that vertical cumulative plot has correct y-axis range."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        ecdf(x=data, ax=ax)

        y_min, y_max = ax.get_ylim()
        assert y_min == pytest.approx(-0.05, abs=0.01)
        assert y_max == pytest.approx(1.05, abs=0.01)
        plt.close(fig)

    def test_horizontal_x_axis_range(self):
        """Test that horizontal cumulative plot has correct x-axis range."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        ecdf(x=data, ax=ax, orientation="horizontal")

        x_min, x_max = ax.get_xlim()
        assert x_min == pytest.approx(-0.05, abs=0.01)
        assert x_max == pytest.approx(1.05, abs=0.01)
        plt.close(fig)

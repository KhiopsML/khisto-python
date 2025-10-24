"""Tests for matplotlib histogram plotting."""

from __future__ import annotations

import numpy as np
import pytest

from khisto.matplotlib import histogram

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt


class TestHistogramBasic:
    """Test basic histogram functionality."""

    def test_simple_array(self):
        """Test histogram with simple numpy array."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax)

        assert container is not None
        # container is a BarContainer when hue is None
        if isinstance(container, list):
            container = container[0]
        assert hasattr(container, "patches")
        assert len(container.patches) > 0
        plt.close(fig)

    def test_without_ax(self):
        """Test histogram without explicit ax parameter."""
        data = np.random.normal(0, 1, 1000)
        container = histogram(x=data)

        assert container is not None
        if isinstance(container, list):
            container = container[0]
        assert len(container.patches) > 0
        plt.close()

    def test_with_title(self):
        """Test histogram with title."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        histogram(x=data, ax=ax, title="Test Histogram")

        assert ax.get_title() == "Test Histogram"
        plt.close(fig)

    def test_horizontal_orientation(self):
        """Test horizontal histogram."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax, orientation="horizontal")

        assert container is not None
        if isinstance(container, list):
            container = container[0]
        assert len(container.patches) > 0
        plt.close(fig)


class TestHistogramWithDataFrame:
    """Test histogram with DataFrame input."""

    def test_pandas_dataframe(self):
        """Test histogram with pandas DataFrame."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B"], 1000),
            }
        )

        fig, ax = plt.subplots()
        containers = histogram(data=df, x="value", ax=ax)

        assert containers is not None
        plt.close(fig)

    def test_pandas_with_hue(self):
        """Test histogram with pandas DataFrame and hue."""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B"], 1000),
            }
        )

        fig, ax = plt.subplots()
        containers = histogram(data=df, x="value", hue="category", ax=ax)

        assert isinstance(containers, list)
        assert len(containers) == 2
        plt.close(fig)

    def test_polars_dataframe(self):
        """Test histogram with polars DataFrame."""
        pl = pytest.importorskip("polars")

        df = pl.DataFrame(
            {
                "value": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B"], 1000),
            }
        )

        fig, ax = plt.subplots()
        containers = histogram(data=df, x="value", ax=ax)

        assert containers is not None
        plt.close(fig)


class TestHistogramStyling:
    """Test histogram styling options."""

    def test_custom_color(self):
        """Test histogram with custom color."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax, color="red")

        if isinstance(container, list):
            container = container[0]
        # Check that bars have the specified color
        assert container.patches[0].get_facecolor()[:3] == (1.0, 0.0, 0.0)
        plt.close(fig)

    def test_alpha(self):
        """Test histogram with alpha transparency."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax, alpha=0.5)

        if isinstance(container, list):
            container = container[0]
        # Check that bars have the specified alpha
        assert container.patches[0].get_alpha() == 0.5
        plt.close(fig)

    def test_edgecolor(self):
        """Test histogram with edge color."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax, edgecolor="black")

        if isinstance(container, list):
            container = container[0]
        # Check that bars have edge color
        assert container.patches[0].get_edgecolor()[:3] == (0.0, 0.0, 0.0)
        plt.close(fig)

    def test_linewidth(self):
        """Test histogram with custom line width."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax, linewidth=2.0, edgecolor="black")

        if isinstance(container, list):
            container = container[0]
        # Check that bars have the specified linewidth
        assert container.patches[0].get_linewidth() == 2.0
        plt.close(fig)


class TestHistogramLabels:
    """Test histogram axis labels."""

    def test_custom_xlabel(self):
        """Test histogram with custom x label."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        histogram(x=data, ax=ax, xlabel="Custom X")

        assert ax.get_xlabel() == "Custom X"
        plt.close(fig)

    def test_custom_ylabel(self):
        """Test histogram with custom y label."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        histogram(x=data, ax=ax, ylabel="Custom Y")

        assert ax.get_ylabel() == "Custom Y"
        plt.close(fig)

    def test_default_labels_vertical(self):
        """Test default labels for vertical histogram."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        histogram(x=data, ax=ax)

        assert ax.get_xlabel() == "_value"
        assert ax.get_ylabel() == "Density"
        plt.close(fig)


class TestHistogramGranularity:
    """Test histogram granularity options."""

    def test_best_granularity(self):
        """Test histogram with best granularity."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax, granularity="best")

        assert container is not None
        if isinstance(container, list):
            container = container[0]
        assert len(container.patches) > 0
        plt.close(fig)

    def test_specific_granularity(self):
        """Test histogram with specific granularity level."""
        data = np.random.normal(0, 1, 1000)
        fig, ax = plt.subplots()
        container = histogram(x=data, ax=ax, granularity=0)

        assert container is not None
        if isinstance(container, list):
            container = container[0]
        assert len(container.patches) > 0
        plt.close(fig)


class TestHistogramErrors:
    """Test error handling."""

    def test_no_data_error(self):
        """Test that error is raised when no data provided."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Either 'data' or 'x' must be provided"):
            histogram(ax=ax)
        plt.close(fig)

    def test_string_x_without_data_error(self):
        """Test that error is raised when x is string but no data."""
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="x must be an array"):
            histogram(x="column_name", ax=ax)
        plt.close(fig)

    def test_no_x_with_data_error(self):
        """Test that error is raised when data provided but no x."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"value": [1, 2, 3]})

        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="x column name must be specified"):
            histogram(data=df, ax=ax)
        plt.close(fig)

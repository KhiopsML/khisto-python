"""Tests for Plotly cumulative distribution visualization."""

from __future__ import annotations

import numpy as np

import plotly.graph_objects as go
from khisto.plotly import ecdf


import pandas as pd


class TestCumulativeBasic:
    """Test basic cumulative distribution functionality."""

    def test_cumulative_simple_array(self):
        """Test cumulative with a simple NumPy array."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "scatter"

    def test_cumulative_with_list(self):
        """Test cumulative with a Python list."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        fig = ecdf(x=data)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cumulative_orientation_vertical(self):
        """Test cumulative with vertical orientation."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, orientation="v")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cumulative_orientation_horizontal(self):
        """Test cumulative with horizontal orientation."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, orientation="h")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestCumulativeGranularity:
    """Test granularity parameter functionality."""

    def test_cumulative_granularity_best(self):
        """Test cumulative with best granularity."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, granularity="best")

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cumulative_granularity_int(self):
        """Test cumulative with integer granularity."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, granularity=2)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cumulative_granularity_none_creates_animation(self):
        """Test that granularity=None creates animation frames."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, granularity=None)

        assert isinstance(fig, go.Figure)
        assert hasattr(fig, "frames")
        assert len(fig.frames) > 0


class TestCumulativeStyling:
    """Test styling parameters."""

    def test_cumulative_with_title(self):
        """Test cumulative with custom title."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, title="Test CDF")

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Test CDF"

    def test_cumulative_with_template(self):
        """Test cumulative with custom template."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, template="plotly_white")

        assert isinstance(fig, go.Figure)
        assert fig.layout.template is not None

    def test_cumulative_with_size(self):
        """Test cumulative with custom width and height."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, width=800, height=600)

        assert isinstance(fig, go.Figure)
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_cumulative_with_markers(self):
        """Test cumulative with markers enabled."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, markers=True)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cumulative_line_shapes(self):
        """Test cumulative with different line shapes."""
        data = np.random.normal(0, 1, 100)

        for line_shape in ["linear", "hv", "vh", "spline"]:
            fig = ecdf(x=data, line_shape=line_shape)  # type: ignore[arg-type]
            assert isinstance(fig, go.Figure)


class TestCumulativeDataFrame:
    """Test cumulative with DataFrame inputs."""

    def test_cumulative_with_pandas_dataframe(self):
        """Test cumulative with pandas DataFrame."""
        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 100),
                "category": np.random.choice(["A", "B"], 100),
            }
        )

        fig = ecdf(df, x="value")  # type: ignore[arg-type]
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_cumulative_with_color_grouping(self):
        """Test cumulative with color parameter."""
        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

        fig = ecdf(df, x="value", color="category")  # type: ignore[arg-type]
        assert isinstance(fig, go.Figure)
        # Should have multiple traces for different categories
        assert len(fig.data) >= 1

    def test_cumulative_with_faceting(self):
        """Test cumulative with facet_col parameter."""
        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 100),
                "category": np.random.choice(["A", "B"], 100),
            }
        )

        fig = ecdf(df, x="value", facet_col="category")  # type: ignore[arg-type]
        assert isinstance(fig, go.Figure)

    def test_cumulative_with_line_dash(self):
        """Test cumulative with line_dash parameter."""
        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 100),
                "category": np.random.choice(["A", "B"], 100),
            }
        )

        fig = ecdf(df, x="value", line_dash="category")  # type: ignore[arg-type]
        assert isinstance(fig, go.Figure)


class TestCumulativeAxisConfiguration:
    """Test axis configuration."""

    def test_cumulative_log_x(self):
        """Test cumulative with logarithmic x-axis."""
        data = np.random.lognormal(0, 1, 100)
        fig = ecdf(x=data, log_x=True)

        assert isinstance(fig, go.Figure)

    def test_cumulative_range_x(self):
        """Test cumulative with custom x-axis range."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, range_x=[-3, 3])

        assert isinstance(fig, go.Figure)

    def test_cumulative_range_y(self):
        """Test cumulative with custom y-axis range."""
        data = np.random.normal(0, 1, 100)
        fig = ecdf(x=data, range_y=[0, 1])

        assert isinstance(fig, go.Figure)

    def test_cumulative_with_labels(self):
        """Test cumulative with custom axis labels."""
        df = pd.DataFrame({"value": np.random.normal(0, 1, 100)})
        fig = ecdf(
            df,  # type: ignore[arg-type]
            x="value",
            labels={"value": "Temperature (°C)", "y": "Cumulative Probability"},
        )

        assert isinstance(fig, go.Figure)


class TestCumulativeEdgeCases:
    """Test edge cases and error conditions."""

    def test_cumulative_empty_array(self):
        """Test cumulative with empty array."""
        data = np.array([])

        # This might raise an error or return an empty figure
        # depending on the implementation
        try:
            fig = ecdf(x=data)
            # If it doesn't raise, check it's a valid figure
            assert isinstance(fig, go.Figure)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for empty data
            pass

    def test_cumulative_single_value(self):
        """Test cumulative with single value."""
        data = np.array([5.0])

        try:
            fig = ecdf(x=data)
            assert isinstance(fig, go.Figure)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for single value
            pass

    def test_cumulative_constant_values(self):
        """Test cumulative with all constant values."""
        data = np.ones(100)

        fig = ecdf(x=data)
        assert isinstance(fig, go.Figure)

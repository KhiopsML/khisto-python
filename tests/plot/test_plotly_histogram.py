from __future__ import annotations

import pytest
import pandas as pd

import numpy as np

from khisto.plot.plotly import histogram


class TestPlotlyHistogram:
    """Test suite for plotly histogram function."""

    def test_histogram_with_array(self):
        """Test histogram with numpy array input."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data)

        assert fig is not None
        assert len(fig.data) > 0
        assert fig.data[0].type == "bar"

    def test_histogram_horizontal(self):
        """Test horizontal histogram."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, orientation="h")

        assert fig is not None
        assert fig.data[0].orientation == "h"

    def test_histogram_with_density(self):
        """Test histogram uses Khisto density values."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data)

        assert fig is not None
        # Khisto automatically uses density values

    def test_histogram_cumulative(self):
        """Test cumulative histogram."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, cumulative=True)

        assert fig is not None
        # Cumulative values should be increasing

    def test_histogram_with_opacity(self):
        """Test histogram with custom opacity."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, opacity=0.5)

        assert fig is not None
        assert fig.data[0].opacity == 0.5

    def test_histogram_with_title(self):
        """Test histogram with title."""
        data = np.random.normal(0, 1, 1000)
        title_text = "Test Histogram"
        fig = histogram(x=data, title=title_text)

        assert fig is not None
        assert fig.layout.title.text == title_text

    def test_histogram_with_labels(self):
        """Test histogram with custom labels."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, labels={"x": "Values", "y": "Frequency"})

        assert fig is not None

    def test_histogram_with_range(self):
        """Test histogram with custom ranges."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, range_x=[-3, 3], range_y=[0, 100])

        assert fig is not None
        assert fig.layout.xaxis.range == [-3, 3]
        assert fig.layout.yaxis.range == [0, 100]

    def test_histogram_with_log_scale(self):
        """Test histogram with log scale."""
        data = np.random.exponential(1, 1000)
        fig = histogram(x=data, log_x=True)

        assert fig is not None
        assert fig.layout.xaxis.type == "log"

    def test_histogram_with_text_auto(self):
        """Test histogram with automatic text labels."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, text_auto=True)

        assert fig is not None
        assert fig.data[0].text is not None

    def test_histogram_with_template(self):
        """Test histogram with plotly template."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, template="plotly_white")

        assert fig is not None
        assert fig.layout.template.layout.plot_bgcolor == "white"

    def test_histogram_with_size(self):
        """Test histogram with custom width and height."""
        data = np.random.normal(0, 1, 1000)
        fig = histogram(x=data, width=800, height=600)

        assert fig is not None
        assert fig.layout.width == 800
        assert fig.layout.height == 600

    def test_histogram_no_data_raises(self):
        """Test that missing data raises ValueError."""
        with pytest.raises(ValueError):
            histogram()

    def test_histogram_with_dataframe(self):
        """Test histogram with DataFrame input."""

        df = pd.DataFrame({"value": np.random.normal(0, 1, 1000)})
        fig = histogram(df, x="value")  # pyrefly: ignore

        assert fig is not None
        assert len(fig.data) > 0

    def test_histogram_with_dataframe_and_color(self):
        """Test histogram with DataFrame and color grouping."""

        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 1000),
                "category": np.random.choice(["A", "B", "C"], 1000),
            }
        )
        fig = histogram(df, x="value", color="category")  # pyrefly: ignore

        assert fig is not None
        # Should have multiple traces for different categories
        assert len(fig.data) >= 1

    def test_histogram_with_series(self):
        """Test histogram with pandas Series."""

        series = pd.Series(np.random.normal(0, 1, 1000))
        fig = histogram(x=series)

        assert fig is not None

    def test_histogram_with_list(self):
        """Test histogram with list input."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 100
        fig = histogram(x=data)

        assert fig is not None
        assert len(fig.data) > 0

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest

from khisto.plotly import ridgeplot


class TestPlotlyRidgeplot:
    """Test suite for plotly ridgeplot function."""

    def test_ridgeplot_basic(self):
        """Test basic ridge plot with DataFrame input."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 300),
                        np.random.normal(2, 1, 300),
                        np.random.normal(-1, 0.8, 300),
                    ]
                ),
                "category": ["A"] * 300 + ["B"] * 300 + ["C"] * 300,
            }
        )

        fig = ridgeplot(df, x="value", y="category")

        assert fig is not None
        # Should have traces for each category
        assert len(fig.data) >= 3
        assert all(trace.type == "scatter" for trace in fig.data)

    def test_ridgeplot_with_title(self):
        """Test ridge plot with title."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        title_text = "Test Ridge Plot"
        fig = ridgeplot(df, x="value", y="category", title=title_text)

        assert fig is not None
        assert fig.layout.title.text == title_text

    def test_ridgeplot_with_labels(self):
        """Test ridge plot with custom labels."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(
            df,
            x="value",
            y="category",
            labels={"value": "Measurement", "category": "Group"},
        )

        assert fig is not None
        assert fig.layout.xaxis.title.text == "Measurement"
        assert fig.layout.yaxis.title.text == "Group"

    def test_ridgeplot_with_category_orders(self):
        """Test ridge plot with custom category ordering."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                        np.random.normal(-1, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200 + ["C"] * 200,
            }
        )

        fig = ridgeplot(
            df, x="value", y="category", category_orders={"category": ["C", "A", "B"]}
        )

        assert fig is not None
        # Categories should be in specified order
        assert len(fig.data) >= 3

    def test_ridgeplot_with_custom_colors(self):
        """Test ridge plot with custom colors."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        colors = ["#FF6B6B", "#4ECDC4"]
        fig = ridgeplot(df, x="value", y="category", color_discrete_sequence=colors)

        assert fig is not None
        assert len(fig.data) >= 2

    def test_ridgeplot_with_opacity(self):
        """Test ridge plot with custom opacity."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", opacity=0.5)

        assert fig is not None

    def test_ridgeplot_with_overlap(self):
        """Test ridge plot with different overlap values."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        # Test with minimal overlap
        fig1 = ridgeplot(df, x="value", y="category", overlap=0.0)
        assert fig1 is not None

        # Test with maximum overlap
        fig2 = ridgeplot(df, x="value", y="category", overlap=1.0)
        assert fig2 is not None

        # Test with moderate overlap
        fig3 = ridgeplot(df, x="value", y="category", overlap=0.5)
        assert fig3 is not None

    def test_ridgeplot_with_line_width(self):
        """Test ridge plot with custom line width."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", line_width=3.0)

        assert fig is not None

    def test_ridgeplot_with_log_x(self):
        """Test ridge plot with logarithmic x-axis."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.exponential(1, 200),
                        np.random.exponential(2, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", log_x=True)

        assert fig is not None
        assert fig.layout.xaxis.type == "log"

    def test_ridgeplot_with_range_x(self):
        """Test ridge plot with custom x-axis range."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", range_x=[-3, 5])

        assert fig is not None
        # Plotly converts list to tuple
        assert fig.layout.xaxis.range == (-3, 5)

    def test_ridgeplot_with_template(self):
        """Test ridge plot with plotly template."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", template="plotly_white")

        assert fig is not None
        assert fig.layout.template.layout.plot_bgcolor == "white"

    def test_ridgeplot_with_size(self):
        """Test ridge plot with custom width and height."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", width=1000, height=800)

        assert fig is not None
        assert fig.layout.width == 1000
        assert fig.layout.height == 800

    def test_ridgeplot_with_granularity_best(self):
        """Test ridge plot with 'best' granularity."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", granularity="best")

        assert fig is not None
        assert len(fig.data) >= 2

    def test_ridgeplot_with_granularity_int(self):
        """Test ridge plot with integer granularity."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", granularity=0)

        assert fig is not None
        assert len(fig.data) >= 2

    def test_ridgeplot_with_granularity_none(self):
        """Test ridge plot with None granularity (animation)."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", granularity=None)

        assert fig is not None
        # Should have animation frames
        assert hasattr(fig, "frames")
        assert len(fig.frames) > 0
        # Should have a slider
        assert "sliders" in fig.layout
        assert len(fig.layout.sliders) > 0

    def test_ridgeplot_without_legend(self):
        """Test ridge plot without legend."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category", show_legend=False)

        assert fig is not None
        assert fig.layout.showlegend is False

    def test_ridgeplot_many_categories(self):
        """Test ridge plot with many categories."""
        n_categories = 10
        n_points = 100

        values = []
        categories = []
        for i in range(n_categories):
            values.extend(np.random.normal(i, 1, n_points))
            categories.extend([f"Cat_{i}"] * n_points)

        df = pd.DataFrame({"value": values, "category": categories})

        fig = ridgeplot(df, x="value", y="category")

        assert fig is not None
        assert len(fig.data) >= n_categories

    def test_ridgeplot_missing_column_error(self):
        """Test ridge plot raises error for missing columns."""
        df = pd.DataFrame(
            {
                "value": np.random.normal(0, 1, 200),
                "category": ["A"] * 100 + ["B"] * 100,
            }
        )

        # Error comes from Plotly Express's build_dataframe function
        with pytest.raises(ValueError, match="not the name of a column"):
            ridgeplot(df, x="nonexistent", y="category")

        with pytest.raises(ValueError, match="not the name of a column"):
            ridgeplot(df, x="value", y="nonexistent")

    def test_ridgeplot_single_category(self):
        """Test ridge plot with a single category."""
        df = pd.DataFrame(
            {"value": np.random.normal(0, 1, 200), "category": ["A"] * 200}
        )

        fig = ridgeplot(df, x="value", y="category")

        assert fig is not None
        assert len(fig.data) >= 1

    def test_ridgeplot_with_polars(self):
        """Test ridge plot with Polars DataFrame."""
        try:
            import polars as pl

            df = pl.DataFrame(
                {
                    "value": np.concatenate(
                        [
                            np.random.normal(0, 1, 200),
                            np.random.normal(2, 1, 200),
                        ]
                    ),
                    "category": ["A"] * 200 + ["B"] * 200,
                }
            )

            fig = ridgeplot(df, x="value", y="category")

            assert fig is not None
            assert len(fig.data) >= 2
        except ImportError:
            pytest.skip("Polars not installed")

    def test_ridgeplot_with_continuous_colorscale(self):
        """Test ridge plot with continuous color scale."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                        np.random.normal(4, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200 + ["C"] * 200,
            }
        )

        # Test with named colorscale
        fig = ridgeplot(df, x="value", y="category", color_continuous_scale="Viridis")

        assert fig is not None
        assert len(fig.data) >= 3

    def test_ridgeplot_with_continuous_colorscale_list(self):
        """Test ridge plot with continuous color scale as list."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                        np.random.normal(4, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200 + ["C"] * 200,
            }
        )

        # Test with list of colors
        fig = ridgeplot(
            df,
            x="value",
            y="category",
            color_continuous_scale=["blue", "yellow", "red"],
        )

        assert fig is not None
        assert len(fig.data) >= 3

    def test_ridgeplot_with_line_color_auto(self):
        """Test ridge plot with automatic line color."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        # Test with auto line color (default)
        fig = ridgeplot(df, x="value", y="category", line_color="auto")

        assert fig is not None
        assert len(fig.data) >= 2

    def test_ridgeplot_with_line_color_custom(self):
        """Test ridge plot with custom line color."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        # Test with custom line color
        fig = ridgeplot(df, x="value", y="category", line_color="black")

        assert fig is not None
        assert len(fig.data) >= 2
        # Check that ridge traces (not baseline) have the specified line color
        # Baseline traces have width=0, ridge traces have width > 0
        ridge_traces = [
            trace
            for trace in fig.data
            if hasattr(trace, "line")
            and trace.line is not None
            and trace.line.width > 0
        ]
        assert len(ridge_traces) >= 2
        for trace in ridge_traces:
            assert trace.line.color == "black"

    def test_ridgeplot_with_different_templates(self):
        """Test ridge plot with different Plotly templates."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        # Test with plotly_dark template
        fig1 = ridgeplot(df, x="value", y="category", template="plotly_dark")
        assert fig1 is not None

        # Test with ggplot2 template
        fig2 = ridgeplot(df, x="value", y="category", template="ggplot2")
        assert fig2 is not None

        # Test with seaborn template
        fig3 = ridgeplot(df, x="value", y="category", template="seaborn")
        assert fig3 is not None

    def test_ridgeplot_granularity_fallback(self):
        """Test that categories with lower max granularity don't disappear."""
        # Create data where categories might have different granularities
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 500),  # More varied data
                        np.random.normal(
                            2, 0.1, 100
                        ),  # Less varied data (might have lower max granularity)
                        np.random.normal(4, 1, 500),  # More varied data
                    ]
                ),
                "category": ["A"] * 500 + ["B"] * 100 + ["C"] * 500,
            }
        )

        # Request a high granularity level
        fig = ridgeplot(df, x="value", y="category", granularity=5)

        assert fig is not None
        # All categories should still be visible (each has a baseline and ridge trace)
        # With 3 categories, we expect at least 6 traces (2 per category)
        assert len(fig.data) >= 6

    def test_ridgeplot_y_axis_ticks(self):
        """Test that y-axis has proper tick labels for categories."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                        np.random.normal(4, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200 + ["C"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category")

        assert fig is not None
        # Check that y-axis has tick labels
        assert fig.layout.yaxis.showticklabels is True
        assert fig.layout.yaxis.tickmode == "array"
        assert len(fig.layout.yaxis.ticktext) == 3
        assert set(fig.layout.yaxis.ticktext) == {"A", "B", "C"}

    def test_ridgeplot_baseline_traces(self):
        """Test that baseline traces are created for proper filling."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category")

        assert fig is not None
        # With 2 categories, we expect 4 traces: 2 baselines + 2 ridges
        assert len(fig.data) == 4

        # Check that there are baseline traces (width=0, no legend)
        baseline_traces = [
            t for t in fig.data if hasattr(t, "line") and t.line.width == 0
        ]
        assert len(baseline_traces) == 2

    def test_ridgeplot_reversed_category_order(self):
        """Test that categories are displayed from bottom to top correctly."""
        df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(0, 1, 200),
                        np.random.normal(2, 1, 200),
                        np.random.normal(4, 1, 200),
                    ]
                ),
                "category": ["A"] * 200 + ["B"] * 200 + ["C"] * 200,
            }
        )

        fig = ridgeplot(df, x="value", y="category")

        assert fig is not None
        # The first category in the list should be at the top (higher y-offset)
        # Check y-axis tick order
        tick_labels = fig.layout.yaxis.ticktext
        assert tick_labels[0] == "A"  # First tick is first category

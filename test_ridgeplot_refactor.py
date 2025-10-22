"""Quick test of the refactored ridgeplot function."""

import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame(
    {
        "value": np.concatenate(
            [
                np.random.normal(0, 1, 500),
                np.random.normal(2, 1.5, 500),
                np.random.normal(-1, 0.8, 500),
            ]
        ),
        "category": ["A"] * 500 + ["B"] * 500 + ["C"] * 500,
    }
)

# Test basic ridgeplot
from khisto.plot.plotly import ridgeplot

try:
    fig = ridgeplot(df, x="value", y="category")
    print("✓ Basic ridgeplot created successfully")

    # Test with custom category order
    fig = ridgeplot(
        df,
        x="value",
        y="category",
        category_orders={"category": ["C", "A", "B"]},
        title="Distribution Comparison",
    )
    print("✓ Ridgeplot with custom category order created successfully")

    # Test with granularity parameter
    fig = ridgeplot(df, x="value", y="category", granularity=3)
    print("✓ Ridgeplot with specific granularity created successfully")

    print("\n✓✓✓ All tests passed! ✓✓✓")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()

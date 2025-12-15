# Khisto

**Optimal Binning Histograms for Python**

Khisto is a Python library for creating histograms and cumulative distribution functions (ECDFs) using the **Khiops optimal binning algorithm**. Unlike standard histograms that use fixed-width bins or simple heuristics, Khisto automatically determines the optimal number of bins and their variable widths to best represent the underlying data distribution.

It provides a core Array API for data analysis and drop-in replacements for **Matplotlib** and **Plotly** plotting functions.

## Features

- **Optimal Binning**: Uses the MODL (Minimum Description Length) principle to find the best discretization.
- **Variable-Width Bins**: Captures dense regions with fine bins and sparse regions with wider bins.
- **Array API Support**: Works with NumPy, PyTorch, JAX, CuPy, and more (via `narwhals` and `array-api-compat`).
- **Plotting Integrations**:
  - **Matplotlib**: `khisto.matplotlib.hist` (drop-in replacement for `plt.hist`)
  - **Plotly**: `khisto.plotly.histogram` (compatible with Plotly Express)

## Installation

### From PyPI

Install the core library:

```bash
pip install khisto
```

Install with plotting support:

```bash
# For Matplotlib support
pip install "khisto[matplotlib]"

# For Plotly support
pip install "khisto[plotly]"

# For everything
pip install "khisto[all]"
```

## Usage

### 1. Core Array API

Compute optimal histograms and ECDFs directly from arrays.

```python
import numpy as np
from khisto.array import histogram, ecdf

# Generate data
data = np.random.normal(0, 1, 1000)

# Compute optimal histogram (returns density values and bin edges)
values, edges = histogram(data, granularity="best")

# Compute ECDF
cdf = ecdf(data)
print(cdf(0.0))  # Evaluate CDF at x=0
```

### 2. Matplotlib Integration

Use Khisto as a drop-in replacement for `plt.hist`.

```python
import matplotlib.pyplot as plt
import khisto.matplotlib as khm
import numpy as np

data = np.random.normal(0, 1, 1000)

# Standard Matplotlib histogram
# plt.hist(data, bins=30)

# Khisto optimal histogram
khm.hist(data, density=True)

plt.show()
```

### 3. Plotly Integration

Create interactive histograms with optimal binning.

```python
import plotly.express as px
import khisto.plotly as khp

df = px.data.tips()

# Create histogram
fig = khp.histogram(df, x="total_bill", color="sex")
fig.show()
```

## Development

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and packaging.

### Installation from Source

1.  **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/khisto-python.git
    cd khisto-python
    ```

3.  **Sync dependencies**:
    Create a virtual environment and install all dependencies (including dev and optional extras):
    ```bash
    uv sync --all-extras
    ```

4.  **Run Tests**:
    ```bash
    uv run pytest
    ```

### Project Structure

- `src/khisto/array`: Core algorithms and Array API.
- `src/khisto/matplotlib`: Matplotlib plotting backends.
- `src/khisto/plotly`: Plotly plotting backends.
- `src/khiops`: C++ backend source (requires CMake to build if modifying).

## License

[BSD 3-Clause Clear License](LICENSE)

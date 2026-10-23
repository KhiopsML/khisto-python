# Khisto

[![CI](https://github.com/KhiopsML/khisto-python/actions/workflows/ci.yaml/badge.svg)](https://github.com/KhiopsML/khisto-python/actions/workflows/ci.yaml)
[![Docs](https://github.com/KhiopsML/khisto-python/actions/workflows/docs.yaml/badge.svg)](https://khiopsml.github.io/khisto-python/)
[![Release](https://img.shields.io/github/v/release/KhiopsML/khisto-python?include_prereleases)](https://github.com/KhiopsML/khisto-python/releases)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)](https://github.com/KhiopsML/khisto-python/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-BSD--3--Clause--Clear-blue)](LICENSE)

**Optimal Binning Histograms for Python**

Khisto is a Python library for creating histograms using the **Khiops optimal binning algorithm**. Unlike standard histograms that use fixed-width bins or simple heuristics, Khisto automatically determines the optimal number of bins and their variable widths to best represent the underlying data distribution.

Documentation is available at **[khiopsml.github.io/khisto-python](https://khiopsml.github.io/khisto-python/)**.

| Standard Gaussian | Heavy-tailed Pareto |
| --- | --- |
| ![Adaptive Gaussian histogram](https://khiopsml.github.io/khisto-python/_images/gaussian-quick-start.png) | ![Adaptive Pareto histogram](https://khiopsml.github.io/khisto-python/_images/pareto-quick-start.png) |

## Installation

```bash
pip install khisto
```

With matplotlib support:

```bash
pip install "khisto[matplotlib]"
```

## Quick Start

```python
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

# Generate 10,000 samples from a Normal distribution
normal_data = np.random.normal(size=10000)

# Plot an adaptive histogram
n, bins, patches = hist(normal_data)
plt.show()

# Generate 10,000 samples from a Pareto distribution
long_tail_data = np.random.pareto(3, size=10000)

# Plot an adaptive histogram on logarithmic axes.
n, bins, patches = hist(long_tail_data)
plt.xscale("symlog")
plt.yscale("log")
plt.show()
```

## Development

On Windows, install [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with the **Desktop development with C++** workload before running the commands
below.

```bash
# Clone repository
git clone https://github.com/KhiopsML/khisto-python.git
cd khisto-python

# Install with dev dependencies
uv sync --group dev --extra all

# Run tests
uv run pytest
```

### Secure release builds

After the October 2026 emSign CA incident, CI uses uv >=0.13.4, pip >=26.2.2,
and certifi >=2026.10.19. A new virtual environment alone is not sufficient:
its installer and build-container certificate bundles must also be patched.
See the [Python packaging advisory](https://discuss.python.org/t/109576).

CI bootstraps pip with patched uv, disables dependency caches, and installs
development dependencies from the committed lockfile with `uv sync --frozen`.
Wheel builds use cibuildwheel 4.3.4 with post-incident, digest-pinned Linux
images. The jobs verify the active TLS clients, test the wheel files themselves,
and check the publisher image's certifi and sigstore before Trusted Publishing
with attestations.

For a recovery release, use a fresh checkout and hosted runners, do not restore
old build artifacts or container layers, and publish a new version: PyPI does
not allow replacing an existing filename. Verify all platform tests, PyPI file
hashes and attestations, an uncached installation, and the deployed documentation
version. These checks establish the new build's provenance; they do not prove
that a previous build or a developer workstation was uncompromised.

## License

[BSD 3-Clause Clear License](LICENSE)

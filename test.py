# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

# %%
import numpy as np
import matplotlib.pyplot as plt
from khisto.matplotlib import hist

# Generate 10,000 samples from a Pareto distribution
long_tail_data = np.random.pareto(3, size=10000)

# Plot an adaptive histogram on logarithmic axes.
n, bins, patches = hist(long_tail_data, density=True)
plt.xscale("symlog")
plt.yscale("log")
plt.show()

# Generate 10,000 samples from a Normal distribution
normal_data = np.random.normal(size=10000)

# Plot an adaptive histogram
n, bins, patches = hist(normal_data, density=True)
plt.show()

# %%

# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

"""Check that the base API works without optional dependencies."""

import importlib.util

import numpy as np

import khisto

assert importlib.util.find_spec("matplotlib") is None
counts, edges = khisto.histogram(
    np.array([1.0, 2.0, 3.0]),
    density=False,
)
assert counts.sum() == 3
assert len(edges) == len(counts) + 1

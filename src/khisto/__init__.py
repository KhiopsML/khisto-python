# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KHISTO_BIN_DIR = os.environ.get("KHISTO_BIN_DIR", "khisto")

# Metadata first: a pyproject.toml check broke installs with a stray one in site-packages.
try:
    __version__ = version("khisto")
except PackageNotFoundError:
    # Source checkout without install: read the repository's pyproject.toml.
    # TODO : Remove on Python 3.10 EOL
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open(Path(__file__).resolve().parents[2] / "pyproject.toml", "rb") as f:
        __version__ = tomllib.load(f)["project"]["version"]

from .core import HistogramResult
from .histogram import histogram
from .matplotlib import hist

__all__ = [
    "HistogramResult",
    "hist",
    "histogram",
]

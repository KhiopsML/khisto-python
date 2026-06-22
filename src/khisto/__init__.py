# Copyright (c) 2025-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

import logging
import os
from importlib.metadata import version
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
KHISTO_BIN_DIR = os.environ.get("KHISTO_BIN_DIR", "khisto")

if (ROOT_DIR / "pyproject.toml").exists():
    # Development mode: package not installed; pyproject.toml present
    # TODO : Remove on Python 3.10 EOL
    try:
        import tomllib as tomli
    except ModuleNotFoundError:
        import tomli

    with open(ROOT_DIR / "pyproject.toml", "rt") as f:
        __version__ = tomli.load(f)["project"]["version"]
else:
    # User mode: package installed; pyproject.toml not directly accessible
    from importlib.metadata import version  # noqa: E402

    __version__ = version("khisto")

from .array import histogram  # noqa: E402
from .core import HistogramResult  # noqa: E402

__all__ = [
    "histogram",
    "HistogramResult",
]

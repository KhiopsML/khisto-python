# Copyright (c) 2023-2026 Orange. All rights reserved.
# This software is distributed under the BSD 3-Clause-clear License, the text of which is available
# at https://spdx.org/licenses/BSD-3-Clause-Clear.html or see the "LICENSE" file for more details.

import os
from importlib.metadata import version
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()

KHISTO_BIN_DIR = os.environ.get("KHISTO_BIN_DIR", "khisto")

__version__ = version("khisto")

from .array import histogram  # noqa: E402
from .core import HistogramResult  # noqa: E402

__all__ = [
    "histogram",
    "HistogramResult",
]

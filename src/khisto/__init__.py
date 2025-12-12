import os
from importlib.metadata import version
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()

KHISTO_BIN_DIR = os.environ.get("KHISTO_BIN_DIR", "khisto")

__version__ = version("khisto")

from .array import (  # noqa: E402
    histogram,
    histogram_bin_edges,
    histogram_table,
    ECDFResult,
    ECDFResultCollection,
    ecdf,
    ecdf_values,
    ecdf_values_table,
)

__all__ = [
    "histogram",
    "histogram_bin_edges",
    "histogram_table",
    "ECDFResult",
    "ECDFResultCollection",
    "ecdf",
    "ecdf_values",
    "ecdf_values_table",
]

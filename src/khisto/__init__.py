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

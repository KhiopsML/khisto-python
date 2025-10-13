import os
from importlib.metadata import version
from pathlib import Path


ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()

__version__ = version("khisto")
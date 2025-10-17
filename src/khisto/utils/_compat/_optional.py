from __future__ import annotations

import importlib
from enum import Enum
from typing import Literal, Optional
from types import ModuleType
from khisto import logger


class Extras(Enum):
    PLOTLY = "plotly"
    MATPLOTLIB = "matplotlib"
    ALL = "all"


def import_optional_dependency(
    name: str,
    extra: Extras = Extras.ALL,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> Optional[ModuleType]:
    """
    Import an optional dependency.

    By default, if a dependency is missing, an ImportError with a descriptive
    message will be raised. If a dependency is present but too old, we raise
    an exception.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when the dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is too old.
          Warn that the version is too old and return None
        * ignore: If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users perform version validation locally when
          using ``errors="ignore"`` (see ``io/html.py``)
    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package version is too old and `errors`
        is ``'warn'``.
    """

    install_name = name

    msg = (
        f"Missing optional dependency '{install_name}'. "
        f"Use: ```pip install khisto[{extra.value}]```."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as e:
        if errors == "raise":
            raise ImportError(msg) from e
        elif errors == "warn":
            print(msg)
            return None
        else:
            return None
    except ModuleNotFoundError as e:  # type: ignore  # noqa: PGH003
        if errors == "raise":
            raise ModuleNotFoundError(msg) from e
        elif errors == "warn":
            logger.warning(msg)
            return None
        else:
            return None

    return module

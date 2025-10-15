from __future__ import annotations
from types import ModuleType
from typing import TYPE_CHECKING
import pyarrow as pa
from array import array
from array_api_compat import array_namespace

if TYPE_CHECKING:
    from khisto.typing import ArrayT


class ListBackend(ModuleType):
    """A simple backend to handle Python lists and tuples."""

    @staticmethod
    def asarray(x: pa.Array) -> list:
        """Convert PyArrow Array to Python list.

        Parameters
        ----------
        x : pa.Array
            PyArrow Array to convert.

        Returns
        -------
        list
            Python list representation of the PyArrow Array.
        """
        return x.to_pylist()


class ArrowBackend(ModuleType):
    """A backend to handle PyArrow arrays."""

    @staticmethod
    def asarray(x: pa.Array) -> pa.Array:
        """Return PyArrow Array unchanged.

        Parameters
        ----------
        x : pa.Array
            PyArrow Array input.

        Returns
        -------
        pa.Array
            The same PyArrow Array.
        """
        return x


def get_array_backend(x: ArrayT) -> ModuleType:
    """Get the appropriate backend for the given array-like input.

    Parameters
    ----------
    x : ArrayT
        Array supporting the Python Array API standard, or a list, tuple, or array.array.

    Returns
    -------
    ModuleType
        Backend module appropriate for the input type. Returns ListBackend for lists,
        tuples, and array.array; ArrowBackend for PyArrow arrays; otherwise returns
        the array namespace from array_api_compat.
    """
    if isinstance(x, (list, tuple, array)):
        backend = ListBackend(name="ListBackend")
    elif isinstance(x, pa.Array):
        backend = ArrowBackend(name="ArrowBackend")
    else:
        backend = array_namespace(x)
    return backend

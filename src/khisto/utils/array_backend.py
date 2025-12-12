from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, List
import pyarrow as pa
from array import array
from array_api_compat import array_namespace
import narwhals as nw

from khisto.utils.narwhals import parse_narwhals_series

if TYPE_CHECKING:
    from khisto.typing import ArrayT


class ListBackend(ModuleType):
    """A simple backend to handle Python lists and tuples."""

    @staticmethod
    def asarray(x: pa.Array | list) -> list:
        """Convert PyArrow Array to Python list.

        Parameters
        ----------
        x : pa.Array or list
            PyArrow Array to convert.

        Returns
        -------
        list
            Python list representation of the PyArrow Array.
        """
        if isinstance(x, list):
            return x
        return x.to_pylist()


class ScalarBackend(ModuleType):
    """A backend to handle scalar values."""

    @staticmethod
    def asarray(x: pa.Array | List) -> float:
        """Convert PyArrow Array to scalar value.

        Parameters
        ----------
        x : pa.Array or list
            PyArrow Array to convert.

        Returns
        -------
        Scalar
            Scalar representation of the PyArrow Array.
        """
        if isinstance(x, list):
            if len(x) == 1:
                return x[0]
            raise ValueError("Expected a single-element list for scalar conversion.")
        pylist = x.to_pylist()
        if len(pylist) == 1:
            return float(pylist[0])  # type: ignore
        raise ValueError(
            "Expected a single-element PyArrow Array for scalar conversion."
        )


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


class SeriesBackend(ModuleType):
    """A backend to handle Narwhals Series."""

    def __init__(self, name: str, narwhals_backend: ModuleType = nw) -> None:
        super().__init__(name)
        self.narwhals_backend = narwhals_backend

    def asarray(self, x: pa.Array) -> nw.Series:
        """Convert PyArrow Array to Narwhals Series.

        Parameters
        ----------
        x : pa.Array
            PyArrow Array input.

        Returns
        -------
        nw.Series
            Narwhals Series representation of the PyArrow Array.
        """
        return nw.new_series(name="data", values=x, backend=self.narwhals_backend)


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
    if isinstance(x, (int, float, complex)):
        backend = ScalarBackend(name="ScalarBackend")
    elif isinstance(x, (list, tuple, array)):
        backend = ListBackend(name="ListBackend")
    elif isinstance(x, pa.Array):
        backend = ArrowBackend(name="ArrowBackend")
    elif parse_narwhals_series(x) is not None:
        backend = SeriesBackend(
            name="SeriesBackend",
            narwhals_backend=nw.get_native_namespace(x),  # type: ignore
        )
    else:
        backend = array_namespace(x)
    return backend

import pyarrow as pa
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def to_arrow(x) -> pa.Array:
    """Convert input to PyArrow Array.

    Parameters
    ----------
    x : array-like
        Input data to convert to PyArrow Array.

    Returns
    -------
    pa.Array
        PyArrow Array representation of the input data.

    Raises
    ------
    pa.ArrowInvalid
        If the input cannot be converted to PyArrow Array using standard
        conversion, falls back to converting via list with float64 type.
    """
    try:
        return pa.array(x)
    except pa.ArrowInvalid:
        return pa.array(x.tolist(), type=pa.float64())

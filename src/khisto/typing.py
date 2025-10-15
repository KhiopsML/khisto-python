from typing import TYPE_CHECKING, Union
from array import array

if TYPE_CHECKING:
    from array_api_compat import Array  # type: ignore[import]

    ArrayT = Union[Array, list, tuple, array]

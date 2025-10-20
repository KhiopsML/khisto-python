from typing import TYPE_CHECKING, Literal, Union
from array import array

if TYPE_CHECKING:
    from array_api_compat import Array  # type: ignore[import]

    ArrayT = Union[Array, list, tuple, array]
    GranularityT = Union[int, Literal["best"]]

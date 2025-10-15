from typing import Optional
import narwhals as nw


def parse_narwhals_series(x) -> Optional[nw.Series]:
    """Check if the input is a Narwhals Series.

    Parameters
    ----------
    x : any
        Input to check for Narwhals Series compatibility.

    Returns
    -------
    Optional[nw.Series]
        A Narwhals Series if the input can be converted, None otherwise.
    """
    try:
        return nw.from_native(x, series_only=True)
    except (TypeError, ValueError):
        return None

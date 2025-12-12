import numpy as np
import pyarrow as pa
from khisto.array import ecdf


def test_ecdf_evaluate_linear_interpolation():
    # Data: 0, 1, 2, 3, 4
    data = [0.0, 1.0, 2.0, 3.0, 4.0]
    cdf = ecdf(data)

    # Get positions and values
    pos = cdf.positions
    if hasattr(pos, "to_pylist"):
        pos = pos.to_pylist()

    vals = cdf.cdf_values
    if hasattr(vals, "to_pylist"):
        vals = vals.to_pylist()

    assert len(pos) == len(vals)
    assert len(pos) >= 2

    # Test interpolation at midpoint of first interval
    p0, p1 = pos[0], pos[1]
    v0, v1 = vals[0], vals[1]
    mid = (p0 + p1) / 2
    expected = v0 + 0.5 * (v1 - v0)

    # Evaluate
    result = cdf(mid)
    # result might be a 0-d array or float depending on backend
    # If input is float, output is float (based on my implementation returning backend.asarray(result))
    # Wait, backend.asarray(pa.Array) returns list if ListBackend.
    # If input was float, prepare_input might have used ListBackend?
    # Let's check what cdf(mid) returns.

    # If result is a list with 1 element, extract it.
    if isinstance(result, list):
        assert len(result) == 1
        result = result[0]
    elif isinstance(result, pa.Array):
        result = result[0].as_py()

    assert np.isclose(result, expected)

    # Test bounds
    res_low = cdf(pos[0] - 1.0)
    if isinstance(res_low, list):
        res_low = res_low[0]
    assert res_low == 0.0

    res_high = cdf(pos[-1] + 1.0)
    if isinstance(res_high, list):
        res_high = res_high[0]
    assert res_high == 1.0

    # Test array input
    x_arr = [pos[0] - 1.0, mid, pos[-1] + 1.0]
    res = cdf(x_arr)
    # Should return list
    assert isinstance(res, list)
    assert len(res) == 3
    assert res[0] == 0.0
    assert np.isclose(res[1], expected)
    assert res[2] == 1.0


def test_ecdf_evaluate_numpy_input():
    data = [0.0, 1.0, 2.0, 3.0, 4.0]
    cdf = ecdf(data)

    pos = cdf.positions
    if hasattr(pos, "to_pylist"):
        pos = pos.to_pylist()

    vals = cdf.cdf_values
    if hasattr(vals, "to_pylist"):
        vals = vals.to_pylist()

    p0, p1 = pos[0], pos[1]
    v0, v1 = vals[0], vals[1]
    mid = (p0 + p1) / 2
    expected = v0 + 0.5 * (v1 - v0)

    x_arr = np.array([pos[0] - 1.0, mid, pos[-1] + 1.0])
    res = cdf(x_arr)

    assert isinstance(res, np.ndarray)
    assert res.shape == x_arr.shape
    assert res[0] == 0.0
    assert np.isclose(res[1], expected)
    assert res[2] == 1.0

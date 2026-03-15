import numpy as np
import pytest

import pytnl._containers
from pytnl._meta import is_dim_guard
from pytnl.containers import NDArray


def _numpy_dtype_for_value_type(value_type: type) -> np.dtype:
    if value_type is bool:
        return np.dtype(np.bool_)
    if value_type is int:
        return np.dtype(np.int64)
    if value_type is float:
        return np.dtype(np.float64)
    if value_type is complex:
        return np.dtype(np.complex128)
    raise TypeError(f"Unsupported value type: {value_type!r}")


ARRAY_TYPES = (
    pytnl._containers.Array_bool,
    pytnl._containers.Array_int,
    pytnl._containers.Array_float,
    pytnl._containers.Array_complex,
)

VECTOR_TYPES = (
    pytnl._containers.Vector_int,
    pytnl._containers.Vector_float,
    pytnl._containers.Vector_complex,
)

STATIC_VECTOR_TYPES = (
    pytnl._containers.StaticVector_1_int,
    pytnl._containers.StaticVector_1_float,
    pytnl._containers.StaticVector_1_complex,
    pytnl._containers.StaticVector_2_int,
    pytnl._containers.StaticVector_2_float,
    pytnl._containers.StaticVector_2_complex,
    pytnl._containers.StaticVector_3_int,
    pytnl._containers.StaticVector_3_float,
    pytnl._containers.StaticVector_3_complex,
)

NDARRAY_PARAMS = (
    (1, int, (7,)),
    (2, int, (3, 5)),
    (3, int, (2, 3, 4)),
    (1, float, (7,)),
    (2, float, (3, 5)),
    (3, float, (2, 3, 4)),
)


def _fill_linear_1d(obj: object, n: int) -> None:
    value_type = obj.ValueType  # type: ignore[attr-defined]
    for i in range(n):
        if value_type is bool:
            obj[i] = (i % 2) == 0  # type: ignore[index]
        elif value_type is int:
            obj[i] = i + 10  # type: ignore[index]
        elif value_type is float:
            obj[i] = i + 0.5  # type: ignore[index]
        elif value_type is complex:
            obj[i] = complex(i + 1, -i - 2)  # type: ignore[index]
        else:
            raise TypeError(f"Unsupported value type: {value_type!r}")


def _first_mutated_value(value_type: type) -> object:
    if value_type is bool:
        return True
    if value_type is int:
        return 123456789
    if value_type is float:
        return 1234.5
    if value_type is complex:
        return complex(11.5, -7.25)
    raise TypeError(f"Unsupported value type: {value_type!r}")


def _second_mutated_value(value_type: type) -> object:
    if value_type is bool:
        return False
    if value_type is int:
        return -321
    if value_type is float:
        return -3.25
    if value_type is complex:
        return complex(-2.5, 3.0)
    raise TypeError(f"Unsupported value type: {value_type!r}")


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_array_buffer_numpy_zero_copy_writable(array_type: type) -> None:
    n = 8
    a = array_type(n)
    _fill_linear_1d(a, n)

    arr_np = np.asarray(a)
    expected_dtype = _numpy_dtype_for_value_type(array_type.ValueType)

    assert arr_np.shape == (n,)
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    v1 = _first_mutated_value(array_type.ValueType)
    arr_np[0] = v1
    assert a[0] == v1

    v2 = _second_mutated_value(array_type.ValueType)
    a[1] = v2
    assert arr_np[1] == v2


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_array_const_view_buffer_readonly(array_type: type) -> None:
    n = 6
    a = array_type(n)
    _fill_linear_1d(a, n)

    const_view = a.getConstView()
    mv = memoryview(const_view)
    arr_np = np.asarray(const_view)

    assert mv.readonly
    assert arr_np.flags["WRITEABLE"] is False
    assert arr_np.shape == (n,)

    assert isinstance(mv.format, str)
    assert mv.ndim == 1
    assert tuple(mv.shape) == (n,)

    with pytest.raises(ValueError):
        arr_np[0] = arr_np[0]


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_vector_buffer_numpy_zero_copy_writable(vector_type: type) -> None:
    n = 9
    v = vector_type(n)
    _fill_linear_1d(v, n)

    arr_np = np.asarray(v)
    expected_dtype = _numpy_dtype_for_value_type(vector_type.ValueType)

    assert arr_np.shape == (n,)
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    v1 = _first_mutated_value(vector_type.ValueType)
    arr_np[2] = v1
    assert v[2] == v1

    v2 = _second_mutated_value(vector_type.ValueType)
    v[3] = v2
    assert arr_np[3] == v2


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_vector_const_view_buffer_readonly(vector_type: type) -> None:
    n = 5
    v = vector_type(n)
    _fill_linear_1d(v, n)

    const_view = v.getConstView()
    mv = memoryview(const_view)
    arr_np = np.asarray(const_view)

    assert mv.readonly
    assert arr_np.flags["WRITEABLE"] is False
    assert arr_np.shape == (n,)

    assert isinstance(mv.format, str)
    assert mv.ndim == 1
    assert tuple(mv.shape) == (n,)

    with pytest.raises(ValueError):
        arr_np[0] = arr_np[0]


@pytest.mark.parametrize("sv_type", STATIC_VECTOR_TYPES)
def test_static_vector_buffer_numpy_zero_copy_writable(sv_type: type) -> None:
    n = sv_type.getSize()
    if sv_type.ValueType is int:
        init = [i + 1 for i in range(n)]
    elif sv_type.ValueType is float:
        init = [i + 0.25 for i in range(n)]
    elif sv_type.ValueType is complex:
        init = [complex(i + 1, i + 2) for i in range(n)]
    else:
        raise TypeError(f"Unsupported value type: {sv_type.ValueType!r}")

    sv = sv_type(init)
    arr_np = np.asarray(sv)
    expected_dtype = _numpy_dtype_for_value_type(sv_type.ValueType)

    assert arr_np.shape == (n,)
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    v1 = _first_mutated_value(sv_type.ValueType)
    arr_np[0] = v1
    assert sv[0] == v1

    v2 = _second_mutated_value(sv_type.ValueType)
    sv[min(1, n - 1)] = v2
    assert arr_np[min(1, n - 1)] == v2


@pytest.mark.parametrize("dim, value_type, shape", NDARRAY_PARAMS)
def test_ndarray_buffer_numpy_shape_strides_dtype_and_zero_copy(dim: int, value_type: type, shape: tuple[int, ...]) -> None:
    assert is_dim_guard(dim)
    a = NDArray[dim, value_type]()
    a.setSizes(*shape)

    # Fill deterministic values
    for idx in np.ndindex(shape):
        if value_type is int:
            val = int(sum((i + 1) * (p + 1) for p, i in enumerate(idx)))
        elif value_type is float:
            val = float(sum((i + 1) * (p + 0.5) for p, i in enumerate(idx)))
        else:
            raise TypeError("NDARRAY_PARAMS only includes int/float for now")
        a[idx] = val

    arr_np = np.asarray(a)
    expected_dtype = _numpy_dtype_for_value_type(value_type)

    assert arr_np.shape == shape
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    # Verify C-contiguous strides
    itemsize = arr_np.dtype.itemsize
    expected_strides = []
    stride = itemsize
    for extent in reversed(shape):
        expected_strides.append(stride)
        stride *= extent
    expected_strides = tuple(reversed(expected_strides))
    assert arr_np.strides == expected_strides

    # Zero-copy check both directions
    first_idx = tuple(0 for _ in shape)
    if value_type is int:
        v1 = 777
        v2 = -55
    else:
        v1 = 77.75
        v2 = -5.5

    arr_np[first_idx] = v1
    assert a[first_idx] == v1

    a[first_idx] = v2
    assert arr_np[first_idx] == v2


@pytest.mark.parametrize("dim, value_type, shape", NDARRAY_PARAMS)
def test_ndarray_const_view_buffer_readonly(dim: int, value_type: type, shape: tuple[int, ...]) -> None:
    assert is_dim_guard(dim)
    a = NDArray[dim, value_type]()
    a.setSizes(*shape)
    a.setValue(1)

    const_view = a.getConstView()
    mv = memoryview(const_view)
    arr_np = np.asarray(const_view)

    assert mv.readonly
    assert arr_np.flags["WRITEABLE"] is False
    assert arr_np.shape == shape
    assert arr_np.flags["C_CONTIGUOUS"]

    first_idx = tuple(0 for _ in shape)

    with pytest.raises((TypeError, ValueError)):
        mv[first_idx] = mv[first_idx]

    with pytest.raises(ValueError):
        arr_np[first_idx] = arr_np[first_idx]

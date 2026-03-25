from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np
import pytest

import pytnl._containers
from pytnl._meta import is_dim_guard
from pytnl.containers import NDArray

type ScalarType = bool | int | float | complex
type Index1D = int
type IndexND = tuple[int, ...]


class _Mutable1DContainer(Protocol):
    def __getitem__(self, index: Index1D, /) -> ScalarType: ...
    def __setitem__(self, index: Index1D, value: ScalarType, /) -> None: ...


class _SizedStaticVector(Protocol):
    ValueType: type

    @staticmethod
    def getSize() -> int: ...
    def __getitem__(self, index: Index1D, /) -> ScalarType: ...
    def __setitem__(self, index: Index1D, value: ScalarType, /) -> None: ...


class _ContainerTypeWithValueType(Protocol):
    ValueType: type

    def __call__(self, n: int, /) -> _Mutable1DContainer: ...


class _StaticVectorType(Protocol):
    ValueType: type

    @staticmethod
    def getSize() -> int: ...
    def __call__(self, init: list[ScalarType], /) -> _SizedStaticVector: ...


def _numpy_dtype_for_value_type(value_type: type) -> np.dtype[Any]:
    if value_type is bool:
        return np.dtype(np.bool_)
    if value_type is int:
        return np.dtype(np.int64)
    if value_type is float:
        return np.dtype(np.float64)
    if value_type is complex:
        return np.dtype(np.complex128)
    raise TypeError(f"Unsupported value type: {value_type!r}")


ARRAY_TYPES = cast(
    tuple[_ContainerTypeWithValueType, ...],
    cast(
        object,
        (
            pytnl._containers.Array_bool,
            pytnl._containers.Array_int,
            pytnl._containers.Array_float,
            pytnl._containers.Array_complex,
        ),
    ),
)

VECTOR_TYPES = cast(
    tuple[_ContainerTypeWithValueType, ...],
    cast(
        object,
        (
            pytnl._containers.Vector_int,
            pytnl._containers.Vector_float,
            pytnl._containers.Vector_complex,
        ),
    ),
)

STATIC_VECTOR_TYPES = cast(
    tuple[_StaticVectorType, ...],
    cast(
        object,
        (
            pytnl._containers.StaticVector_1_int,
            pytnl._containers.StaticVector_1_float,
            pytnl._containers.StaticVector_1_complex,
            pytnl._containers.StaticVector_2_int,
            pytnl._containers.StaticVector_2_float,
            pytnl._containers.StaticVector_2_complex,
            pytnl._containers.StaticVector_3_int,
            pytnl._containers.StaticVector_3_float,
            pytnl._containers.StaticVector_3_complex,
        ),
    ),
)

NDARRAY_PARAMS = (
    (1, int, (7,)),
    (2, int, (3, 5)),
    (3, int, (2, 3, 4)),
    (1, float, (7,)),
    (2, float, (3, 5)),
    (3, float, (2, 3, 4)),
)


def _fill_linear_1d(obj: _Mutable1DContainer, n: int, value_type: type) -> None:
    for i in range(n):
        if value_type is bool:
            obj[i] = (i % 2) == 0
        elif value_type is int:
            obj[i] = i + 10
        elif value_type is float:
            obj[i] = i + 0.5
        elif value_type is complex:
            obj[i] = complex(i + 1, -i - 2)
        else:
            raise TypeError(f"Unsupported value type: {value_type!r}")


def _first_mutated_value(value_type: type) -> ScalarType:
    if value_type is bool:
        return True
    if value_type is int:
        return 123456789
    if value_type is float:
        return 1234.5
    if value_type is complex:
        return complex(11.5, -7.25)
    raise TypeError(f"Unsupported value type: {value_type!r}")


def _second_mutated_value(value_type: type) -> ScalarType:
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
def test_array_buffer_numpy_zero_copy_writable(array_type: _ContainerTypeWithValueType) -> None:
    n = 8
    value_type = array_type.ValueType
    a = array_type(n)
    _fill_linear_1d(a, n, value_type)

    arr_np = np.asarray(a)
    expected_dtype = _numpy_dtype_for_value_type(value_type)

    assert arr_np.shape == (n,)
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    v1 = _first_mutated_value(value_type)
    arr_np[0] = v1
    assert a[0] == v1

    v2 = _second_mutated_value(value_type)
    a[1] = v2
    assert arr_np[1] == v2


@pytest.mark.parametrize("array_type", ARRAY_TYPES)
def test_array_const_view_buffer_readonly(array_type: _ContainerTypeWithValueType) -> None:
    n = 6
    value_type = array_type.ValueType
    a = array_type(n)
    _fill_linear_1d(a, n, value_type)

    const_view = cast(Any, a).getConstView()
    mv = memoryview(const_view)
    arr_np = np.asarray(const_view)

    assert mv.readonly
    assert arr_np.flags["WRITEABLE"] is False
    assert arr_np.shape == (n,)

    assert isinstance(mv.format, str)
    assert mv.ndim == 1
    assert mv.shape is not None
    assert tuple(mv.shape) == (n,)

    with pytest.raises(ValueError):
        arr_np[0] = arr_np[0]


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_vector_buffer_numpy_zero_copy_writable(vector_type: _ContainerTypeWithValueType) -> None:
    n = 9
    value_type = vector_type.ValueType
    v = vector_type(n)
    _fill_linear_1d(v, n, value_type)

    arr_np = np.asarray(v)
    expected_dtype = _numpy_dtype_for_value_type(value_type)

    assert arr_np.shape == (n,)
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    v1 = _first_mutated_value(value_type)
    arr_np[2] = v1
    assert v[2] == v1

    v2 = _second_mutated_value(value_type)
    v[3] = v2
    assert arr_np[3] == v2


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_vector_const_view_buffer_readonly(vector_type: _ContainerTypeWithValueType) -> None:
    n = 5
    value_type = vector_type.ValueType
    v = vector_type(n)
    _fill_linear_1d(v, n, value_type)

    const_view = cast(Any, v).getConstView()
    mv = memoryview(const_view)
    arr_np = np.asarray(const_view)

    assert mv.readonly
    assert arr_np.flags["WRITEABLE"] is False
    assert arr_np.shape == (n,)

    assert isinstance(mv.format, str)
    assert mv.ndim == 1
    assert mv.shape is not None
    assert tuple(mv.shape) == (n,)

    with pytest.raises(ValueError):
        arr_np[0] = arr_np[0]


@pytest.mark.parametrize("sv_type", STATIC_VECTOR_TYPES)
def test_static_vector_buffer_numpy_zero_copy_writable(sv_type: _StaticVectorType) -> None:
    n = sv_type.getSize()
    value_type = sv_type.ValueType

    if value_type is int:
        init: list[ScalarType] = [i + 1 for i in range(n)]
    elif value_type is float:
        init = [i + 0.25 for i in range(n)]
    elif value_type is complex:
        init = [complex(i + 1, i + 2) for i in range(n)]
    else:
        raise TypeError(f"Unsupported value type: {value_type!r}")

    sv = sv_type(init)
    arr_np = np.asarray(sv)
    expected_dtype = _numpy_dtype_for_value_type(value_type)

    assert arr_np.shape == (n,)
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    v1 = _first_mutated_value(value_type)
    arr_np[0] = v1
    assert sv[0] == v1

    v2 = _second_mutated_value(value_type)
    idx = min(1, n - 1)
    sv[idx] = v2
    assert arr_np[idx] == v2


@pytest.mark.parametrize("dim, value_type, shape", NDARRAY_PARAMS)
def test_ndarray_buffer_numpy_shape_strides_dtype_and_zero_copy(dim: int, value_type: type, shape: tuple[int, ...]) -> None:
    assert is_dim_guard(dim)
    a = NDArray[dim, value_type]()  # type: ignore[index]
    a.setSizes(*shape)

    # Fill deterministic values
    for idx in np.ndindex(shape):
        if value_type is int:
            val_i = int(sum((i + 1) * (p + 1) for p, i in enumerate(idx)))
            a[idx] = val_i
        elif value_type is float:
            a_float = cast(Any, a)
            val_f = float(sum((i + 1) * (p + 0.5) for p, i in enumerate(idx)))
            a_float[idx] = val_f
        else:
            raise TypeError("NDARRAY_PARAMS only includes int/float for now")

    arr_np = np.asarray(a)
    expected_dtype = _numpy_dtype_for_value_type(value_type)

    assert arr_np.shape == shape
    assert arr_np.dtype == expected_dtype
    assert arr_np.flags["C_CONTIGUOUS"]
    assert arr_np.flags["WRITEABLE"]

    # Verify C-contiguous strides
    itemsize = arr_np.dtype.itemsize
    expected_strides_list: list[int] = []
    stride = itemsize
    for extent in reversed(shape):
        expected_strides_list.append(stride)
        stride *= extent
    expected_strides: tuple[int, ...] = tuple(reversed(expected_strides_list))
    assert arr_np.strides == expected_strides

    # Zero-copy check both directions
    first_idx: IndexND = tuple(0 for _ in shape)

    if value_type is int:
        v1_i = 777
        v2_i = -55
        arr_np[first_idx] = v1_i
        assert a[first_idx] == v1_i
        a[first_idx] = v2_i
        assert arr_np[first_idx] == v2_i
    elif value_type is float:
        a_float = cast(Any, a)
        v1_f = 77.75
        v2_f = -5.5
        arr_np[first_idx] = v1_f
        assert a_float[first_idx] == v1_f
        a_float[first_idx] = v2_f
        assert arr_np[first_idx] == v2_f
    else:
        raise TypeError("NDARRAY_PARAMS only includes int/float for now")


@pytest.mark.parametrize("dim, value_type, shape", NDARRAY_PARAMS)
def test_ndarray_const_view_buffer_readonly(dim: int, value_type: type, shape: tuple[int, ...]) -> None:
    assert is_dim_guard(dim)
    a = NDArray[dim, value_type]()  # type: ignore[index]
    a.setSizes(*shape)
    a.setValue(1)

    const_view = a.getConstView()
    mv = memoryview(const_view)  # type: ignore[arg-type]
    arr_np = np.asarray(const_view)

    assert mv.readonly
    assert arr_np.flags["WRITEABLE"] is False
    assert arr_np.shape == shape
    assert arr_np.flags["C_CONTIGUOUS"]

    first_idx: IndexND = tuple(0 for _ in shape)

    with pytest.raises((TypeError, ValueError)):
        mv[first_idx] = mv[first_idx]

    with pytest.raises(ValueError):
        arr_np[first_idx] = arr_np[first_idx]

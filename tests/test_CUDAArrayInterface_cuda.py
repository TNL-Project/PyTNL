# mypy: disable-error-code="import-not-found, no-any-unimported, no-untyped-call, unused-ignore"
# pyright: standard
# pyright: reportMissingImports=information

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    import pytnl._containers_cuda as _containers_cuda
else:
    _containers_cuda = pytest.importorskip("pytnl._containers_cuda")


pytestmark = pytest.mark.cuda


class SupportsCUDAArrayInterface(Protocol):
    @property
    def __cuda_array_interface__(self) -> dict[str, object]: ...


def _typestr_for_bound_class_name(class_name: str) -> str:
    if "_bool" in class_name:
        return "|b1"
    if "_int" in class_name:
        return np.dtype(np.int_).str
    if "_float" in class_name:
        return np.dtype(np.float64).str
    if "_complex" in class_name:
        return np.dtype(np.complex128).str
    raise TypeError(f"Unsupported CAI test class: {class_name!r}")


def _assert_cai_common(
    obj: SupportsCUDAArrayInterface,
    *,
    expected_shape: tuple[int, ...],
    expected_read_only: bool,
    expected_strides: tuple[int, ...] | None,
) -> dict[str, object]:
    cai = obj.__cuda_array_interface__

    assert isinstance(cai, dict)
    assert cai["version"] == 3
    assert cai["shape"] == expected_shape
    assert cai["typestr"] == _typestr_for_bound_class_name(type(obj).__name__)

    data = cai["data"]
    assert isinstance(data, tuple)
    assert len(data) == 2
    assert isinstance(data[0], int)
    assert data[1] is expected_read_only

    assert cai["strides"] == expected_strides
    assert "mask" not in cai or cai["mask"] is None
    assert "stream" in cai
    assert cai["stream"] is None
    return cai


def _make_zero_size_ndarray_2_int() -> SupportsCUDAArrayInterface:
    array = _containers_cuda.NDArray_2_int()
    array.setSizes(0, 5)
    return cast(SupportsCUDAArrayInterface, array)


@pytest.mark.parametrize(
    "array_type",
    (
        _containers_cuda.Array_bool,
        _containers_cuda.Array_int,
        _containers_cuda.Array_float,
        _containers_cuda.Array_complex,
    ),
)
def test_array_cai_fields_and_read_only_flag(array_type: type[Any]) -> None:
    size = 7
    array = array_type(size)
    const_view = array.getConstView()

    _assert_cai_common(
        array,
        expected_shape=(size,),
        expected_read_only=False,
        expected_strides=None,
    )
    _assert_cai_common(
        const_view,
        expected_shape=(size,),
        expected_read_only=True,
        expected_strides=None,
    )


@pytest.mark.parametrize(
    "vector_type",
    (
        _containers_cuda.Vector_int,
        _containers_cuda.Vector_float,
        _containers_cuda.Vector_complex,
    ),
)
def test_vector_cai_fields_and_read_only_flag(vector_type: type[Any]) -> None:
    size = 9
    vector = vector_type(size)
    const_view = vector.getConstView()

    _assert_cai_common(
        vector,
        expected_shape=(size,),
        expected_read_only=False,
        expected_strides=None,
    )
    _assert_cai_common(
        const_view,
        expected_shape=(size,),
        expected_read_only=True,
        expected_strides=None,
    )


@pytest.mark.parametrize(
    "ndarray_type, shape",
    (
        (_containers_cuda.NDArray_1_int, (7,)),
        (_containers_cuda.NDArray_2_int, (3, 5)),
        (_containers_cuda.NDArray_3_int, (2, 3, 4)),
        (_containers_cuda.NDArray_2_float, (4, 3)),
        (_containers_cuda.NDArray_3_complex, (2, 2, 3)),
    ),
)
def test_ndarray_cai_shape_strides_and_read_only_flag(ndarray_type: type[Any], shape: tuple[int, ...]) -> None:
    array = ndarray_type()
    array.setSizes(*shape)
    const_view = array.getConstView()

    itemsize = np.dtype(ndarray_type.ValueType).itemsize
    expected_element_strides = tuple(array.getStrides())
    expected_byte_strides = tuple(s * itemsize for s in expected_element_strides)

    _assert_cai_common(
        array,
        expected_shape=shape,
        expected_read_only=False,
        expected_strides=expected_byte_strides,
    )
    _assert_cai_common(
        const_view,
        expected_shape=shape,
        expected_read_only=True,
        expected_strides=expected_byte_strides,
    )


@pytest.mark.parametrize(
    "factory, expected_shape",
    (
        (lambda: _containers_cuda.Array_int(0), (0,)),
        (lambda: _containers_cuda.Vector_float(0), (0,)),
        (_make_zero_size_ndarray_2_int, (0, 5)),
    ),
)
def test_zero_size_cai_uses_null_pointer(
    factory: Callable[[], SupportsCUDAArrayInterface],
    expected_shape: tuple[int, ...],
) -> None:
    obj = factory()
    strides = cast(tuple[int, ...] | None, obj.__cuda_array_interface__["strides"])
    cai = _assert_cai_common(
        obj,
        expected_shape=expected_shape,
        expected_read_only=False,
        expected_strides=None if len(expected_shape) == 1 else strides,
    )

    # CAI v3: zero-size arrays should export pointer value 0.
    data = cast(tuple[int, bool], cai["data"])
    assert data[0] == 0

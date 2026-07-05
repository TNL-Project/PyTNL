# mypy: disable-error-code="import-not-found, no-any-unimported, unused-ignore, attr-defined, operator, misc, arg-type, union-attr"
# pyright: standard
# pyright: reportMissingImports=information
# pyright: reportAttributeAccessIssue=none
# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none
# pyright: reportGeneralTypeIssues=none, reportArgumentType=none, reportAttributeAccessIssue=none
# pyright: reportCallIssue=none, reportFunctionMemberAccess=none

import itertools

import pytest

pytestmark = pytest.mark.cuda

pytest.importorskip("numba_cuda_mlir.cuda")

from numba_cuda_mlir import cuda as nb_cuda  # noqa: E402

import pytnl.containers  # noqa: E402
import pytnl.devices  # noqa: E402
from pytnl.cuda_kernels import CompiledDeviceFunc, compile_device_func  # noqa: E402


def _launch_kernel(kernel: object, sizes: tuple[int, ...], array_args: list[object]) -> None:
    dim = len(sizes)
    block_size = {1: 256, 2: 16, 3: 8}.get(dim, 8)
    grid_dims = tuple((s + block_size - 1) // block_size for s in sizes)
    if dim == 1:
        grid = (grid_dims[0], 1, 1)
        block = (block_size, 1, 1)
    elif dim == 2:
        grid = (grid_dims[1], grid_dims[0], 1)
        block = (block_size, block_size, 1)
    else:
        grid = (grid_dims[dim - 1], grid_dims[dim - 2], grid_dims[dim - 3])
        block = (block_size, block_size, block_size)
    kernel[grid, block](sizes, *array_args)  # type: ignore[index]
    nb_cuda.synchronize()


def test_decorator_bare() -> None:
    """@compile_device_func (bare decorator) returns CompiledDeviceFunc."""

    @compile_device_func
    def setter(*idx: int) -> None:
        pass

    assert isinstance(setter, CompiledDeviceFunc)
    assert setter.name == "setter"


def test_decorator_named() -> None:
    """@compile_device_func(name=...) returns CompiledDeviceFunc with custom name."""

    @compile_device_func(name="my_func")
    def setter(*idx: int) -> None:
        pass

    assert isinstance(setter, CompiledDeviceFunc)
    assert setter.name == "my_func"


def test_forall_via_ndarray_method() -> None:
    """NDArray.forAll(setter) with CompiledDeviceFunc via CAI capture."""

    a = pytnl.containers.NDArray[3, float, pytnl.devices.Cuda]()  # type: ignore[index]
    a.setSizes(2, 3, 4)
    a.setValue(0.0)

    @compile_device_func
    def setter(*idx: int) -> None:
        a[idx] = 77.0

    a.forAll(setter)

    for idx in itertools.product(range(2), range(3), range(4)):
        assert a[idx] == 77.0


def test_forall_interior_boundary_via_ndarray_methods() -> None:
    """NDArray.forInterior/forBoundary with CompiledDeviceFunc via CAI capture."""

    a = pytnl.containers.NDArray[2, float, pytnl.devices.Cuda]()  # type: ignore[index]
    a.setSizes(5, 5)
    a.setValue(0.0)

    @compile_device_func
    def setter(*idx: int) -> None:
        a[idx] = 55.0

    a.forInterior(setter)

    for i in range(5):
        for j in range(5):
            if 0 < i < 4 and 0 < j < 4:
                assert a[i, j] == 55.0, f"interior ({i},{j})"
            else:
                assert a[i, j] == 0.0, f"boundary ({i},{j})"

    a2 = pytnl.containers.NDArray[2, float, pytnl.devices.Cuda]()  # type: ignore[index]
    a2.setSizes(5, 5)
    a2.setValue(0.0)

    @compile_device_func
    def setter2(*idx: int) -> None:
        a2[idx] = 55.0

    a2.forBoundary(setter2)

    for i in range(5):
        for j in range(5):
            if 0 < i < 4 and 0 < j < 4:
                assert a2[i, j] == 0.0, f"interior ({i},{j})"
            else:
                assert a2[i, j] == 55.0, f"boundary ({i},{j})"


def test_forall_high_dim_cupy_4d() -> None:
    """forAll with 4D cupy array captured via CAI (3D grid + 1 sequential loop)."""
    cupy = pytest.importorskip("cupy")

    shape = (2, 3, 4, 5)
    arr = cupy.zeros(shape, dtype=cupy.float64)

    @compile_device_func
    def setter(*idx: int) -> None:
        arr[idx] = 42.0

    kernel, array_arg_names = setter.get_kernel(4, "all")
    _launch_kernel(kernel, shape, setter.resolve_array_args(array_arg_names))

    assert cupy.all(arr == 42.0)


def test_forall_high_dim_cupy_5d() -> None:
    """forAll with 5D cupy array captured via CAI (3D grid + 2 sequential loops)."""
    cupy = pytest.importorskip("cupy")

    shape = (2, 3, 2, 3, 2)
    arr = cupy.zeros(shape, dtype=cupy.float64)

    @compile_device_func
    def setter(*idx: int) -> None:
        arr[idx] = 7.0

    kernel, array_arg_names = setter.get_kernel(5, "all")
    _launch_kernel(kernel, shape, setter.resolve_array_args(array_arg_names))

    assert cupy.all(arr == 7.0)


def test_forinterior_high_dim_cupy_4d() -> None:
    """forInterior with 4D cupy array captured via CAI."""
    cupy = pytest.importorskip("cupy")

    shape = (3, 4, 5, 6)
    arr = cupy.zeros(shape, dtype=cupy.float64)

    @compile_device_func
    def setter(*idx: int) -> None:
        arr[idx] = 99.0

    kernel, array_arg_names = setter.get_kernel(4, "interior")
    _launch_kernel(kernel, shape, setter.resolve_array_args(array_arg_names))

    for idx in itertools.product(*(range(s) for s in shape)):
        is_interior = all(1 <= i < s - 1 for i, s in zip(idx, shape))
        if is_interior:
            assert arr[idx] == 99.0, f"interior {idx}"
        else:
            assert arr[idx] == 0.0, f"boundary {idx}"


def test_forboundary_high_dim_raises() -> None:
    """forBoundary with dim > 3 raises ValueError (matches TNL C++ static_assert)."""

    @compile_device_func
    def setter(*idx: int) -> None:
        pass

    with pytest.raises(ValueError, match="forBoundary is not supported for dimension > 3"):
        setter.get_kernel(4, "boundary")

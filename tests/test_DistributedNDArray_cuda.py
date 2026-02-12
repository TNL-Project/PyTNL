# mypy: disable-error-code="import-not-found, no-any-unimported, no-untyped-call, unused-ignore"
# pyright: standard
# pyright: reportMissingImports=information

import copy
import math
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any, cast

import mpi4py
import mpi4py.MPI
import numpy as np
import pytest

from pytnl._meta import DIMS, VT, is_dim_guard
from pytnl.containers import DistributedNDArray
from pytnl.devices import Cuda

if TYPE_CHECKING:
    import pytnl._containers_cuda as _containers_cuda
else:
    _containers_cuda = pytest.importorskip("pytnl._containers_cuda")

# Mark all tests in this module
pytestmark = pytest.mark.cuda

# Type alias for view types
type DistributedNDArrayView = (
    _containers_cuda.DistributedNDArrayView_1_complex
    | _containers_cuda.DistributedNDArrayView_2_complex
    | _containers_cuda.DistributedNDArrayView_3_complex
    | _containers_cuda.DistributedNDArrayView_1_int
    | _containers_cuda.DistributedNDArrayView_2_int
    | _containers_cuda.DistributedNDArrayView_3_int
    | _containers_cuda.DistributedNDArrayView_1_float
    | _containers_cuda.DistributedNDArrayView_2_float
    | _containers_cuda.DistributedNDArrayView_3_float
)
type DistributedNDArrayView_const = (
    _containers_cuda.DistributedNDArrayView_1_complex
    | _containers_cuda.DistributedNDArrayView_2_complex_const
    | _containers_cuda.DistributedNDArrayView_3_complex_const
    | _containers_cuda.DistributedNDArrayView_1_int_const
    | _containers_cuda.DistributedNDArrayView_2_int_const
    | _containers_cuda.DistributedNDArrayView_3_int_const
    | _containers_cuda.DistributedNDArrayView_1_float_const
    | _containers_cuda.DistributedNDArrayView_2_float_const
    | _containers_cuda.DistributedNDArrayView_3_float_const
)

# number of MPI ranks
NPROC = mpi4py.MPI.COMM_WORLD.Get_size()

# shapes of local NDArrays
GLOBAL_SHAPE_PARAMS = [
    (NPROC * 3,),
    (NPROC * 3, 4),
    (NPROC * 3, 4, 5),
    # TODO: only up to 3D have bindings for now
    # (3, 4, 5, 6),
    # (3, 4, 5, 6, 7),
    # (3, 4, 5, 6, 7, 8),
]

TYPEDEFS_PARAMS = [
    (1, int),
    (2, int),
    (3, int),
    (1, float),
    (2, float),
    (3, float),
    (1, complex),
    (2, complex),
    (3, complex),
]


def distribute_along_x(
    shape: tuple[int, ...],
    rank: int = mpi4py.MPI.COMM_WORLD.rank,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Returns a pair of tuples representing the `[begin, end)` range of indices
    for a local NDArray given a global shape and an MPI rank.

    Parameters:
        shape (tuple[int, ...]): The global shape of the array.
        rank (int): The MPI rank.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: A pair of tuples representing
            the `[begin, end)` range of indices for a local NDArray.
    """
    local_begin = [0] * len(shape)
    local_end = list(shape)

    local_begin[0] = rank * shape[0] // NPROC

    if rank == NPROC - 1:
        local_end[0] = shape[0]
    else:
        local_end[0] = (rank + 1) * shape[0] // NPROC

    return tuple(local_begin), tuple(local_end)


def ndindex_range(local_begin: tuple[int, ...], local_end: tuple[int, ...]) -> Generator[tuple[int, ...]]:
    """
    Wrapper around `np.ndindex` which iterates over global indices in the `[begin, end)` range.
    """
    local_shape = tuple(np.array(local_end) - np.array(local_begin))
    for local_idx in np.ndindex(local_shape):
        # shift local index to global index
        idx = tuple(np.array(local_idx) + np.array(local_begin))
        yield idx


def test_pythonization() -> None:
    assert DistributedNDArray[1, int, Cuda] is _containers_cuda.DistributedNDArray_1_int
    assert DistributedNDArray[2, float, Cuda] is _containers_cuda.DistributedNDArray_2_float
    assert DistributedNDArray[3, complex, Cuda] is _containers_cuda.DistributedNDArray_3_complex


@pytest.mark.parametrize("dim, value_type", TYPEDEFS_PARAMS)
def test_typedefs(dim: DIMS, value_type: type[VT]) -> None:
    """
    Tests the static `IndexerType` and `ValueType` typedefs of the
    DistributedNDArray class.

    Verifies:
    - `ValueType` matches the expected Python type (e.g., `int`, `float`, `bool`).
    - `ViewType` and `ConstViewType` have the same dimension and `ValueType`
    - `LocalViewType` and `ConstLocalViewType` match the expected Python types.
    """
    ndarray_class = DistributedNDArray[dim, value_type, Cuda]  # type: ignore[type-arg,valid-type]

    # Test ValueType
    assert ndarray_class.ValueType is value_type

    # Test ViewType and ConstViewType
    view = cast(DistributedNDArrayView, ndarray_class.ViewType)
    assert view.getDimension() == dim
    assert view.ValueType is value_type
    const_view = cast(DistributedNDArrayView_const, ndarray_class.ConstViewType)
    assert const_view.getDimension() == dim
    assert const_view.ValueType is value_type


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_setSizes(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)
    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    assert a.getSizes() == shape
    # size is 0 before allocation
    assert a.getLocalStorageSize() == 0


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_setDistribution(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()

    assert a.getSizes() == shape
    assert a.getLocalBegins() == local_begin
    assert a.getLocalEnds() == local_end

    local_size = math.prod(end - begin for begin, end in zip(local_begin, local_end))
    assert a.getLocalStorageSize() == local_size


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_data_access(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()

    a.setValue(42)
    for idx in ndindex_range(local_begin, local_end):
        assert a[idx] == 42


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_invalid_indices(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()

    a.setValue(42)
    for idx in ndindex_range(local_begin, local_end):
        low = tuple(-i - 1 for i in idx)
        high = tuple(i + s for i, s in zip(idx, shape))
        # __getitem__
        with pytest.raises(IndexError):
            a[low]
        with pytest.raises(IndexError):
            a[high]
        # __setitem__
        with pytest.raises(IndexError):
            a[low] = 0
        with pytest.raises(IndexError):
            a[high] = 0


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_setLike(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()

    b = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    b.setLike(a)  # pyright: ignore[reportArgumentType]

    assert b.getSizes() == shape


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_reset(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()

    assert a.getLocalStorageSize() > 0
    a.reset()
    assert a.getSizes() == (0,) * dim
    assert a.getLocalStorageSize() == 0


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_equality(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    # Create first array
    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()
    a.setValue(0)

    # Create second array
    b = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    b.setLike(a)  # pyright: ignore[reportArgumentType]
    b.setValue(0)

    assert a == b, "Arrays with the same shape and value should be equal"

    # Change the first array
    ndidx = ndindex_range(local_begin, local_end)
    idx = next(ndidx)
    a[idx] = 1
    assert a != b, "Arrays with same shape but different values should not be equal"

    # Change the second array to match the first array
    b[idx] = 1
    assert a == b, "Arrays with the same shape and value should be equal"

    # Change the second array
    idx = next(ndidx)
    b[idx] = 2
    assert a != b, "Arrays with same shape but different values should not be equal"

    # Change the first array to match the first array
    a[idx] = 2
    assert a == b, "Arrays with the same shape and value should be equal"


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_getStorageArrayView(shape: tuple[int, ...]) -> None:
    """
    Tests the `getStorageArrayView()` method of the NDArray class.

    Verifies:
    - The storage array is of the correct size and shape.
    - The underlying storage is shared with the NDArray (i.e., modifying one affects the other).
    - Both const and non-const access behave as expected in Python.
    """
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()
    a.setValue(0)  # Initialize all elements to 0

    # Get the internal storage array
    storage = a.getLocalView().getStorageArrayView()

    # 1. Check that storage has the correct size
    assert storage.getSize() == math.prod(end - begin for begin, end in zip(local_begin, local_end)), "Storage array size mismatch"

    # 2. Modify storage array directly and verify NDArray reflects the change
    storage.setValue(1)
    for idx in ndindex_range(local_begin, local_end):
        assert a[idx] == 1, f"Element at {idx} was not updated to 1"
    assert all(storage[i] == 1 for i in range(storage.getSize())), "Storage array not fully updated"

    # 3. Modify NDArray and verify storage reflects the change
    a.setValue(2)
    for idx in ndindex_range(local_begin, local_end):
        assert a[idx] == 2, f"Element at {idx} was not updated to 2"
    assert all(storage[i] == 2 for i in range(storage.getSize())), "Storage array not updated from NDArray"

    # 4. Test reference behavior: update a single element via storage
    idx = local_begin
    storage[0] = 99
    assert a[idx] == 99, "Storage array is not a reference to NDArray data"

    # 5. Test reference behavior: update via NDArray and check storage
    a[idx] = 42
    assert storage[a.getStorageIndex(*idx)] == 42, "NDArray is not a reference to storage array data"


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_getView(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()
    a.setValue(0)  # Initialize all elements to 0

    v = a.getView()

    # Check that v has the correct size and shape
    assert v.getSizes() == shape, "View array size mismatch"

    # Check that v reflects changes in a and vice versa
    for idx in ndindex_range(local_begin, local_end):
        assert v[idx] == 0, f"Element at {idx} in view was not initialized to 0"
        v[idx] = idx[0] + idx[-1]
        assert v[idx] == idx[0] + idx[-1], f"Element at {idx} in view was not updated correctly"
        assert a[idx] == idx[0] + idx[-1], "View array is not a reference to NDArray data"

    # Check that storage views are equal
    try:
        assert list(a.getLocalView().getStorageArrayView()) == list(v.getLocalView().getStorageArrayView()), "Storage views are not equal"
    except NotImplementedError:
        pytest.xfail(reason="The __iter__ method is not implemented yet for GPU arrays.")


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_getConstView(shape: tuple[int, ...]) -> None:
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()
    a.setValue(0)  # Initialize all elements to 0

    v = a.getConstView()

    # Check that v has the correct size and shape
    assert v.getSizes() == shape, "View array size mismatch"

    # Check that v reflects changes in a and vice versa
    for idx in ndindex_range(local_begin, local_end):
        assert v[idx] == 0, f"Element at {idx} in view was not initialized to 0"
        # Check that const view cannot be modified directly
        with pytest.raises(TypeError):
            v[idx] = 1

    # Check that storage views are equal
    try:
        assert list(a.getConstLocalView().getStorageArrayView()) == list(v.getConstLocalView().getStorageArrayView()), "Storage views are not equal"
    except NotImplementedError:
        pytest.xfail(reason="The __iter__ method is not implemented yet for GPU arrays.")


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
@pytest.mark.parametrize("copy_function", [copy.copy, copy.deepcopy])
def test_copy(shape: tuple[int, ...], copy_function: Callable[[Any], Any]) -> None:
    """
    Tests the `__copy__` and `__deepcopy__` methods of the NDArray class.
    """
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    a = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    a.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    a.setDistribution(local_begin, local_end)
    a.allocate()
    a.setValue(0)

    b = copy_function(a)

    # 1. Check shape and values match
    assert b.getSizes() == a.getSizes() == shape, "Shape mismatch in shallow copy"
    for idx in ndindex_range(local_begin, local_end):
        assert b[idx] == a[idx], f"Value mismatch at index {idx} in shallow copy"
    assert a == b

    # 2. Modify original, check that copy stays the same
    ndidx = ndindex_range(local_begin, local_end)
    idx = next(ndidx)
    a[idx] = 42
    assert b[idx] == 0
    assert a != b

    # 3. Modify copy, check that original stays the same
    idx = next(ndidx)
    b[idx] = 99
    assert a[idx] == 0
    assert a != b


# Test parameters: (value_type, shape)
STR_REPR_TEST_PARAMS = [
    (int, (3,)),
    (int, (3, 4)),
    (int, (2, 3, 4)),
    (float, (5,)),
    (float, (5, 6)),
    (float, (2, 4, 5)),
    (complex, (5,)),
    (complex, (5, 6)),
    (complex, (2, 4, 5)),
    (int, (0,)),  # Empty 1D array
    (int, (0, 0)),  # Empty 2D array
    (int, (0, 0, 0)),  # Empty 3D array
]


@pytest.mark.parametrize("value_type, shape", STR_REPR_TEST_PARAMS)
def test_str_repr(value_type: type[VT], shape: tuple[int, ...]) -> None:
    """
    Tests the `__str__` and `__repr__` methods of the NDArray class.

    Verifies:
    - The `__str__` returns a readable string with value type and shape.
    - The `__repr__` returns a unique, unambiguous representation with memory address.
    """
    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    # Create NDArray with the given value type and dimension
    array_type = DistributedNDArray[dim, value_type, Cuda]()  # type: ignore[index]
    array_type.setSizes(*shape)

    # Check that `__str__` contains the correct value type and shape
    str_output = str(array_type)
    expected_str = f"DistributedNDArray[{dim}, {value_type.__name__}, Cuda]({', '.join(str(x) for x in shape)})"
    assert str_output == expected_str

    # Check that `__repr__` includes memory address
    repr_output = repr(array_type)
    assert f"NDArray_{dim}_{value_type.__name__} object at 0x" in repr_output
    assert repr_output.endswith(">")


@pytest.mark.parametrize("shape", GLOBAL_SHAPE_PARAMS)
def test_dlpack(shape: tuple[int, ...]) -> None:
    """
    Tests interoperability with CuPy using the DLPack API.

    Verifies:
    - The returned CuPy array has the correct shape and dtype.
    - The array contains the same data as the NDArray.
    - The underlying memory is shared.
    - Changes in CuPy are reflected in the NDArray and vice versa.
    """
    if TYPE_CHECKING:
        import cupy  # type: ignore[import-untyped] # NOQA: PLC0415
    else:
        cupy = pytest.importorskip("cupy")

    dim = len(shape)
    # dim needs to be narrowed down to a literal for type-checking
    assert is_dim_guard(dim)

    # Create and initialize the NDArray
    array = DistributedNDArray[dim, int, Cuda]()  # type: ignore[index]
    array.setSizes(*shape)
    local_begin, local_end = distribute_along_x(shape)
    array.setDistribution(local_begin, local_end)
    array.allocate()
    array.setValue(42)  # Fill with known value

    # Convert to CuPy array
    array_cupy = cupy.from_dlpack(array.getLocalView())

    # Check that the array is writable
    # FIXME: CuPy does not have the writeable flag yet https://github.com/cupy/cupy/issues/2616
    # assert array_cupy.flags.writeable

    # Check shape
    local_shape = tuple(end - begin for begin, end in zip(local_begin, local_end))
    assert array_cupy.shape == local_shape, f"Expected shape {local_shape}, got {array_cupy.shape}"

    # Check strides (CuPy is in bytes, TNL in elements)
    strides = tuple(s // array_cupy.dtype.itemsize for s in array_cupy.strides)
    assert array.getLocalView().getStrides() == strides

    # Check data type
    assert array_cupy.dtype == cupy.int_, f"Expected dtype {cupy.int_}, got {array_cupy.dtype}"

    # Check element-wise equality
    assert cupy.all(array_cupy == 42), "Data mismatch in CuPy array"

    # Test storage array
    storage = array.getLocalView().getStorageArrayView()
    storage_cupy = cupy.from_dlpack(storage)
    assert storage_cupy.shape == (storage.getSize(),)
    assert cupy.all(storage_cupy == 42), "Storage array as_numpy() mismatch"

    # Check that memory is shared
    assert cupy.shares_memory(array_cupy, storage_cupy), "Memory should be shared between NDArray and its storage Array"

    ndidx = cupy.ndindex(local_shape)

    # Modify CuPy array and verify NDArray reflects the change
    idx = next(ndidx)
    array_cupy[idx] = 99
    assert array.getLocalView()[idx] == 99, "CuPy array modification not reflected in NDArray"

    # Modify NDArray and verify CuPy array reflects the change
    idx = next(ndidx)
    array.getLocalView()[idx] = 77
    assert array_cupy[idx] == 77, "NDArray modification not reflected in CuPy array"

    # Check that memory is shared
    assert cupy.shares_memory(array_cupy, cupy.from_dlpack(array.getLocalView())), "Memory should be shared between two CuPy arrays"

    # Get CuPy array from view
    view = array.getView()
    view_cupy = cupy.from_dlpack(view.getLocalView())
    assert view_cupy.shape == local_shape, f"Expected shape {local_shape}, got {view_cupy.shape}"
    assert view_cupy.dtype == array_cupy.dtype
    assert cupy.all(view_cupy == array_cupy), "Data mismatch in CuPy array from view"

    # Get CuPy array from const view
    const_view = array.getConstView()
    const_view_cupy = cupy.from_dlpack(const_view.getConstLocalView())
    assert const_view_cupy.shape == local_shape, f"Expected shape {local_shape}, got {const_view_cupy.shape}"
    assert const_view_cupy.dtype == array_cupy.dtype
    assert cupy.all(const_view_cupy == array_cupy), "Data mismatch in CuPy array from const view"

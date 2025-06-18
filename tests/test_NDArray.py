import copy

import numpy as np
import pytest

from pytnl.containers import NDArray

SHAPE_PARAMS = [
    (3,),
    (3, 4),
    (3, 4, 5),
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
]


@pytest.mark.parametrize("dim, value_type", TYPEDEFS_PARAMS)
def test_typedefs(dim, value_type):
    """
    Tests the static `IndexerType` and `ValueType` typedefs of the NDArray class.

    Verifies:
    - `IndexerType.getDimension()` returns the correct dimension.
    - `ValueType` matches the expected Python type (e.g., `int`, `float`, `bool`).
    - `IndexerType` is a valid NDArrayIndexer class for the specified dimension.
    """
    ndarray_class = NDArray[dim, value_type]

    # Test IndexerType
    indexer_type = ndarray_class.IndexerType
    assert isinstance(indexer_type, type)
    assert indexer_type.getDimension() == dim
    assert issubclass(ndarray_class, indexer_type)

    # Test ValueType
    assert ndarray_class.ValueType is value_type

    # Test IndexType
    assert indexer_type.IndexType is int
    assert ndarray_class.IndexType is int


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_setSizes(shape):
    dim = len(shape)
    a = NDArray[dim, int]()
    a.setSizes(shape)
    assert a.getSizes() == shape
    a.setSizes(*shape)
    assert a.getSizes() == shape


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_data_access(shape):
    dim = len(shape)
    a = NDArray[dim, int]()
    a.setSizes(shape)
    a.setValue(42)
    for idx in np.ndindex(shape):
        assert a[idx] == 42


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_invalid_indices(shape):
    dim = len(shape)
    a = NDArray[dim, int]()
    a.setSizes(shape)
    a.setValue(42)
    for idx in np.ndindex(shape):
        low = tuple(-i - 1 for i in idx)
        high = tuple(i + s for i, s in zip(idx, shape))
        # __getitem__
        with pytest.raises(IndexError):
            a[low] == 42
        with pytest.raises(IndexError):
            a[high] == 42
        # __setitem__
        with pytest.raises(IndexError):
            a[low] = 0
        with pytest.raises(IndexError):
            a[high] = 0


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_setLike(shape):
    dim = len(shape)
    a = NDArray[dim, int]()
    a.setSizes(shape)
    b = NDArray[dim, int]()
    b.setLike(a)
    assert b.getSizes() == shape


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_reset(shape):
    dim = len(shape)
    a = NDArray[dim, int]()
    a.setSizes(shape)
    a.reset()
    assert a.getSizes() == (0,) * dim
    assert a.getStorageSize() == 0


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_equality(shape):
    dim = len(shape)
    # Create first array
    a = NDArray[dim, int]()
    a.setSizes(shape)
    a.setValue(0)

    # Create second array
    b = NDArray[dim, int]()
    b.setLike(a)
    b.setValue(0)

    assert a == b, "Arrays with the same shape and value should be equal"

    # Change the first array
    ndidx = np.ndindex(shape)
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


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_forAll(shape):
    a = NDArray[len(shape), int]()
    a.setSizes(*shape)
    a.setValue(0)

    def setter(*idx):
        a[idx] += 1

    a.forAll(setter)

    assert all(value == 1 for value in a.getStorageArray())
    for idx in np.ndindex(shape):
        assert a[idx] == 1


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_forInterior(shape):
    a = NDArray[len(shape), int]()
    a.setSizes(*shape)
    a.setValue(0)

    def setter(*idx):
        a[idx] += 1

    a.forInterior(setter)

    for idx in np.ndindex(shape):
        # Check if interior
        is_interior = all(1 <= i < s - 1 for i, s in zip(idx, shape))
        if is_interior:
            assert a[idx] == 1
        else:
            assert a[idx] == 0


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_forBoundary(shape):
    a = NDArray[len(shape), int]()
    a.setSizes(*shape)
    a.setValue(0)

    def setter(*idx):
        a[idx] += 1

    a.forBoundary(setter)

    for idx in np.ndindex(shape):
        is_boundary = any(i == 0 or i == s - 1 for i, s in zip(idx, shape))
        if is_boundary:
            assert a[idx] == 1
        else:
            assert a[idx] == 0


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_getStorageArray(shape):
    """
    Tests the `getStorageArray()` method of the NDArray class.

    Verifies:
    - The storage array is of the correct size and shape.
    - The underlying storage is shared with the NDArray (i.e., modifying one affects the other).
    - Both const and non-const access behave as expected in Python.
    """
    dim = len(shape)
    a = NDArray[dim, int]()
    a.setSizes(*shape)
    a.setValue(0)  # Initialize all elements to 0

    # Get the internal storage array
    storage = a.getStorageArray()

    # 1. Check that storage has the correct size
    assert storage.getSize() == np.prod(shape), "Storage array size mismatch"

    # 2. Modify storage array directly and verify NDArray reflects the change
    storage.setValue(1)
    for idx in np.ndindex(shape):
        assert a[idx] == 1, f"Element at {idx} was not updated to 1"
    assert all(storage[i] == 1 for i in range(storage.getSize())), "Storage array not fully updated"

    # 3. Modify NDArray and verify storage reflects the change
    a.setValue(2)
    for idx in np.ndindex(shape):
        assert a[idx] == 2, f"Element at {idx} was not updated to 2"
    assert all(storage[i] == 2 for i in range(storage.getSize())), "Storage array not updated from NDArray"

    # 4. Test reference behavior: update a single element via storage
    idx = (0,) * dim
    storage[0] = 99
    assert a[idx] == 99, "Storage array is not a reference to NDArray data"

    # 5. Test reference behavior: update via NDArray and check storage
    idx = (1,) * dim
    a[idx] = 42
    assert storage[a.getStorageIndex(*idx)] == 42, "NDArray is not a reference to storage array data"


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
@pytest.mark.parametrize("copy_function", [copy.copy, copy.deepcopy])
def test_copy(shape, copy_function):
    """
    Tests the `__copy__` and `__deepcopy__` methods of the NDArray class.
    """
    dim = len(shape)
    a = NDArray[dim, int]()
    a.setSizes(*shape)
    a.setValue(0)

    b = copy_function(a)

    # 1. Check shape and values match
    assert b.getSizes() == a.getSizes() == shape, "Shape mismatch in shallow copy"
    for idx in np.ndindex(shape):
        assert b[idx] == a[idx], f"Value mismatch at index {idx} in shallow copy"
    assert a == b

    # 2. Modify original, check that copy stays the same
    idx = (0,) * dim
    a[idx] = 42
    assert b[idx] == 0
    assert a != b

    # 3. Modify copy, check that original stays the same
    idx = (1,) * dim
    b[idx] = 99
    assert a[idx] == 0
    assert a != b


@pytest.mark.parametrize("shape", SHAPE_PARAMS)
def test_as_numpy(shape):
    """
    Tests the `as_numpy()` method of the NDArray class.

    Verifies:
    - The returned NumPy array has the correct shape and dtype.
    - The array contains the same data as the NDArray.
    - The underlying memory is shared.
    - Changes in NumPy are reflected in the NDArray and vice versa.
    """

    # Create and initialize the NDArray
    array = NDArray[len(shape), int]()
    array.setSizes(*shape)
    array.setValue(42)  # Fill with known value

    # Convert to NumPy array
    array_np = array.as_numpy()

    # Check shape
    assert array_np.shape == shape, f"Expected shape {shape}, got {array_np.shape}"

    # Check strides (NumPy is in bytes, TNL in elements)
    strides = tuple(s // array_np.dtype.itemsize for s in array_np.strides)
    assert array.getStrides() == strides

    # Check data type
    assert array_np.dtype == np.int_, f"Expected dtype {np.int_}, got {array_np.dtype}"

    # Check element-wise equality
    assert np.all(array_np == 42), "Data mismatch in NumPy array"

    # Test storage array
    storage = array.getStorageArray()
    storage_np = storage.as_numpy()
    assert storage_np.shape == (storage.getSize(),)
    assert np.all(storage_np == 42), "Storage array as_numpy() mismatch"

    # Check that memory is shared
    assert np.shares_memory(array_np, storage_np), "Memory should be shared between NDArray and its storage Array"

    ndidx = np.ndindex(shape)

    # Modify NumPy array and verify NDArray reflects the change
    idx = next(ndidx)
    array_np[idx] = 99
    assert array[idx] == 99, "NumPy array modification not reflected in NDArray"

    # Modify NDArray and verify NumPy array reflects the change
    idx = next(ndidx)
    array[idx] = 77
    assert array_np[idx] == 77, "NDArray modification not reflected in NumPy array"

    # Check that memory is shared
    assert np.shares_memory(array_np, array.as_numpy()), "Memory should be shared between NDArray and NumPy array"


# Test parameters: (value_type, shape)
STR_REPR_TEST_PARAMS = [
    (int, (3,)),
    (int, (3, 4)),
    (int, (2, 3, 4)),
    (float, (5,)),
    (float, (5, 6)),
    (float, (2, 4, 5)),
    (int, (0,)),  # Empty 1D array
    (int, (0, 0)),  # Empty 2D array
    (int, (0, 0, 0)),  # Empty 3D array
]


@pytest.mark.parametrize("value_type, shape", STR_REPR_TEST_PARAMS)
def test_str_repr(value_type, shape):
    """
    Tests the `__str__` and `__repr__` methods of the NDArray class.

    Verifies:
    - The `__str__` returns a readable string with value type and shape.
    - The `__repr__` returns a unique, unambiguous representation with memory address.
    """
    # Create NDArray with the given value type and dimension
    dim = len(shape)
    array_type = NDArray[dim, value_type]()
    array_type.setSizes(*shape)

    # Check that `__str__` contains the correct value type and shape
    str_output = str(array_type)
    expected_str = f"NDArray[{dim}, {value_type.__name__}]({', '.join(str(x) for x in shape)})"
    assert str_output == expected_str

    # Check that `__repr__` includes memory address
    repr_output = repr(array_type)
    assert f"NDArray_{dim}_{value_type.__name__} object at 0x" in repr_output
    assert repr_output.endswith(">")

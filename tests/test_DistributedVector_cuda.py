# mypy: disable-error-code="import-not-found, no-any-unimported, no-untyped-call, unused-ignore"
# pyright: standard
# pyright: reportMissingImports=information

import copy
from typing import TYPE_CHECKING, Any

import mpi4py
import mpi4py.MPI
import pytest

from pytnl._meta import VT
from pytnl.containers import DistributedVector, Vector
from pytnl.devices import Cuda

if TYPE_CHECKING:
    import pytnl._containers_cuda as _containers_cuda
else:
    _containers_cuda = pytest.importorskip("pytnl._containers_cuda")

pytestmark = pytest.mark.cuda

NPROC = mpi4py.MPI.COMM_WORLD.Get_size()
RANK = mpi4py.MPI.COMM_WORLD.Get_rank()
GLOBAL_SIZE = 3 * NPROC + 1
VALUE_TYPES: list[type[VT]] = [int, float, complex]
LOCAL_VIEW_TYPES: dict[type[VT], type[Any]] = {
    int: _containers_cuda.VectorView_int,
    float: _containers_cuda.VectorView_float,
    complex: _containers_cuda.VectorView_complex,
}
CONST_LOCAL_VIEW_TYPES: dict[type[VT], type[Any]] = {
    int: _containers_cuda.VectorView_int_const,
    float: _containers_cuda.VectorView_float_const,
    complex: _containers_cuda.VectorView_complex_const,
}


def local_range() -> tuple[int, int]:
    begin = RANK * GLOBAL_SIZE // NPROC
    end = (RANK + 1) * GLOBAL_SIZE // NPROC
    return begin, end


def test_pythonization_and_inheritance() -> None:
    assert DistributedVector[int, Cuda] is _containers_cuda.DistributedVector_int
    assert DistributedVector[float, Cuda] is _containers_cuda.DistributedVector_float
    assert DistributedVector[complex, Cuda] is _containers_cuda.DistributedVector_complex
    assert issubclass(_containers_cuda.DistributedVector_int, _containers_cuda.DistributedArray_int)
    assert issubclass(_containers_cuda.DistributedVector_float, _containers_cuda.DistributedArray_float)
    assert issubclass(_containers_cuda.DistributedVector_complex, _containers_cuda.DistributedArray_complex)
    vector = DistributedVector[int, Cuda]()
    assert isinstance(vector, _containers_cuda.DistributedVector_int)
    assert isinstance(vector, _containers_cuda.DistributedArray_int)


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_typedefs_and_local_views(value_type: type[VT]) -> None:
    begin, end = local_range()
    local_size = end - begin
    vector_type = DistributedVector[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    assert vector_type.RealType is value_type
    assert vector_type.LocalViewType is LOCAL_VIEW_TYPES[value_type]
    assert vector_type.ConstLocalViewType is CONST_LOCAL_VIEW_TYPES[value_type]

    vector = vector_type()
    vector.setDistribution(begin, end, 0, GLOBAL_SIZE)
    vector.setValue(value_type(5))
    local_view = vector.getLocalView()
    const_local_view = vector.getConstLocalView()
    assert isinstance(local_view, LOCAL_VIEW_TYPES[value_type])
    assert isinstance(const_local_view, CONST_LOCAL_VIEW_TYPES[value_type])
    assert local_view.getSize() == local_size
    assert const_local_view.getSize() == local_size
    for i in range(local_size):
        assert local_view[i] == value_type(5)
        assert const_local_view[i] == value_type(5)
    local_view[0] = value_type(7)
    assert vector[begin] == value_type(7)
    assert const_local_view[0] == value_type(7)


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_distribution_and_lifetime(value_type: type[VT]) -> None:
    begin, end = local_range()
    vector = DistributedVector[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    assert vector.empty()
    assert vector.getSize() == 0

    vector.setDistribution(begin, end, 0, GLOBAL_SIZE)
    assert not vector.empty()
    assert vector.getLocalRange() == (begin, end)
    assert vector.getGhosts() == 0
    assert vector.getSize() == GLOBAL_SIZE

    vector.reset()
    assert vector.empty()
    assert vector.getSize() == 0


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_data_access(value_type: type[VT]) -> None:
    begin, end = local_range()
    vector = DistributedVector[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    vector.setDistribution(begin, end, 0, GLOBAL_SIZE)
    vector.setValue(value_type(42))
    for i in range(begin, end):
        assert vector.getElement(i) == value_type(42)
        assert vector[i] == value_type(42)

    vector.setElement(begin, value_type(7))
    assert vector.getElement(begin) == value_type(7)
    assert vector[begin] == value_type(7)
    vector[begin] = value_type(9)
    assert vector.getElement(begin) == value_type(9)
    assert vector[begin] == value_type(9)


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_invalid_distribution_and_indices(value_type: type[VT]) -> None:
    for begin, end, ghosts, error_type in [
        (-1, 1, 0, ValueError),
        (0, 1, -1, ValueError),
        (0, GLOBAL_SIZE + 1, 0, IndexError),
    ]:
        vector = DistributedVector[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
        with pytest.raises(error_type):
            vector.setDistribution(begin, end, ghosts, GLOBAL_SIZE)

    begin, end = local_range()
    vector = DistributedVector[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    vector.setDistribution(begin, end, 0, GLOBAL_SIZE)
    invalid_indices = [-1, GLOBAL_SIZE]
    if NPROC > 1:
        invalid_indices.append(end % GLOBAL_SIZE)
    for i in invalid_indices:
        with pytest.raises(IndexError):
            vector.getElement(i)
        with pytest.raises(IndexError):
            vector[i]
        with pytest.raises(IndexError):
            vector.setElement(i, value_type(0))
        with pytest.raises(IndexError):
            vector[i] = value_type(0)


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_copy_operations(value_type: type[VT]) -> None:
    begin, end = local_range()
    vector_type = DistributedVector[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    vector = vector_type()
    vector.setDistribution(begin, end, 2, GLOBAL_SIZE)
    vector.setValue(value_type(11))

    set_like = vector_type()
    set_like.setLike(vector)  # pyright: ignore[reportArgumentType]
    set_like.setValue(value_type(11))
    assigned = vector_type()
    assigned.assign(vector)
    shallow_copy = copy.copy(vector)
    deep_copy = copy.deepcopy(vector)
    for other in (set_like, assigned, shallow_copy, deep_copy):
        assert other.getLocalRange() == (begin, end)
        assert other.getGhosts() == 2
        assert other.getSize() == GLOBAL_SIZE
        for i in range(begin, end):
            assert other[i] == value_type(11)
    vector[begin] = value_type(99)
    for other in (assigned, shallow_copy, deep_copy):
        assert other[begin] == value_type(11)

    global_vector = Vector[value_type, Cuda](GLOBAL_SIZE)  # type: ignore[type-arg, valid-type]
    for i in range(GLOBAL_SIZE):
        global_vector[i] = value_type(i + 1)
    from_global = vector_type()
    from_global.setDistribution(begin, end, 0, GLOBAL_SIZE)
    from_global.copyFromGlobal(global_vector.getConstView())
    for i in range(begin, end):
        assert from_global[i] == value_type(i + 1)


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_vector_inplace_arithmetic(value_type: type[VT]) -> None:
    begin, end = local_range()
    vector_type = DistributedVector[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    left = vector_type()
    right = vector_type()
    left.setDistribution(begin, end, 0, GLOBAL_SIZE)
    right.setDistribution(begin, end, 0, GLOBAL_SIZE)
    left.setValue(value_type(12))
    right.setValue(value_type(3))

    left += right
    for i in range(begin, end):
        assert left[i] == value_type(15)
    left -= right
    for i in range(begin, end):
        assert left[i] == value_type(12)
    left *= right
    for i in range(begin, end):
        assert left[i] == value_type(36)
    left /= right
    for i in range(begin, end):
        assert left[i] == value_type(12)


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_scalar_inplace_arithmetic(value_type: type[VT]) -> None:
    begin, end = local_range()
    vector = DistributedVector[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    vector.setDistribution(begin, end, 0, GLOBAL_SIZE)
    vector.setValue(value_type(12))
    scalar = value_type(3)

    vector += scalar
    for i in range(begin, end):
        assert vector[i] == value_type(15)
    vector -= scalar
    for i in range(begin, end):
        assert vector[i] == value_type(12)
    vector *= scalar
    for i in range(begin, end):
        assert vector[i] == value_type(36)
    vector /= scalar
    for i in range(begin, end):
        assert vector[i] == value_type(12)


def test_integer_modulo() -> None:
    begin, end = local_range()
    vector = DistributedVector[int, Cuda]()
    divisor = DistributedVector[int, Cuda]()
    vector.setDistribution(begin, end, 0, GLOBAL_SIZE)
    divisor.setDistribution(begin, end, 0, GLOBAL_SIZE)
    vector.setValue(-22)
    divisor.setValue(5)

    vector %= divisor
    for i in range(begin, end):
        assert vector[i] == -2
    vector.setValue(-22)
    vector %= 5
    for i in range(begin, end):
        assert vector[i] == -2


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_equality(value_type: type[VT]) -> None:
    begin, end = local_range()
    vector_type = DistributedVector[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    left = vector_type()
    right = vector_type()
    left.setDistribution(begin, end, 0, GLOBAL_SIZE)
    right.setDistribution(begin, end, 0, GLOBAL_SIZE)
    left.setValue(value_type(42))
    right.setValue(value_type(42))
    assert left == right

    left[begin] = value_type(1)
    assert left != right
    right[begin] = value_type(1)
    assert left == right


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_synchronization(value_type: type[VT]) -> None:
    begin, end = local_range()
    local_size = end - begin
    left = (RANK - 1) % NPROC
    right = (RANK + 1) % NPROC
    vector = DistributedVector[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    vector.setDistribution(begin, end, 2, GLOBAL_SIZE)
    vector.setValue(value_type(RANK + 1))
    vector.setSynchronizer(1)
    vector.startSynchronization()
    vector.waitForSynchronization()

    view = vector.getLocalViewWithGhosts()
    const_view = vector.getConstLocalViewWithGhosts()
    expected = [value_type(RANK + 1)] * local_size + [value_type(left + 1), value_type(right + 1)]
    assert view.getSize() == len(expected)
    assert const_view.getSize() == len(expected)
    for i, value in enumerate(expected):
        assert view[i] == value
        assert const_view[i] == value


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_str(value_type: type[VT]) -> None:
    begin, end = local_range()
    vector = DistributedVector[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    vector.setDistribution(begin, end, 0, GLOBAL_SIZE)
    expected = f"DistributedVector[{value_type.__name__}, Cuda]({GLOBAL_SIZE}, localRange=({begin}, {end}), ghosts=0)"
    assert str(vector) == expected

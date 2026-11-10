import copy
from typing import TypeVar

import mpi4py
import mpi4py.MPI
import pytest

import pytnl._containers
from pytnl import devices
from pytnl.containers import DistributedArray, DistributedVector

pytestmark = pytest.mark.mpi

NPROC = mpi4py.MPI.COMM_WORLD.Get_size()
RANK = mpi4py.MPI.COMM_WORLD.Get_rank()
GLOBAL_SIZE = 3 * NPROC + 1

V = TypeVar(
    "V",
    pytnl._containers.DistributedVector_int,
    pytnl._containers.DistributedVector_float,
    pytnl._containers.DistributedVector_complex,
)
VECTOR_TYPES = V.__constraints__
VALUE_TYPES = [int, float, complex]
ARRAY_TYPES = (
    pytnl._containers.DistributedArray_int,
    pytnl._containers.DistributedArray_float,
    pytnl._containers.DistributedArray_complex,
)


def get_local_range() -> tuple[int, int]:
    begin = RANK * GLOBAL_SIZE // NPROC
    end = (RANK + 1) * GLOBAL_SIZE // NPROC
    return begin, end


def create_vector(vector_type: type[V], ghosts: int = 0) -> V:
    vector = vector_type()
    begin, end = get_local_range()
    vector.setDistribution(begin, end, ghosts, GLOBAL_SIZE)
    return vector


def assert_local_values(vector: V, expected: int) -> None:
    begin, end = get_local_range()
    for i in range(begin, end):
        assert vector.getElement(i) == expected
        assert vector[i] == expected


def test_pythonization() -> None:
    for value_type, vector_type in zip(VALUE_TYPES, VECTOR_TYPES):
        assert DistributedVector[value_type] is vector_type
        assert DistributedVector[value_type, devices.Host] is vector_type


def test_typedefs() -> None:
    assert pytnl._containers.DistributedVector_int.RealType is int
    assert pytnl._containers.DistributedVector_int.LocalViewType is pytnl._containers.VectorView_int
    assert pytnl._containers.DistributedVector_int.ConstLocalViewType is pytnl._containers.VectorView_int_const

    assert pytnl._containers.DistributedVector_float.RealType is float
    assert pytnl._containers.DistributedVector_float.LocalViewType is pytnl._containers.VectorView_float
    assert pytnl._containers.DistributedVector_float.ConstLocalViewType is pytnl._containers.VectorView_float_const

    assert pytnl._containers.DistributedVector_complex.RealType is complex
    assert pytnl._containers.DistributedVector_complex.LocalViewType is pytnl._containers.VectorView_complex
    assert pytnl._containers.DistributedVector_complex.ConstLocalViewType is pytnl._containers.VectorView_complex_const


def test_inheritance() -> None:
    for value_type, vector_type, array_type in zip(VALUE_TYPES, VECTOR_TYPES, ARRAY_TYPES):
        assert issubclass(vector_type, array_type)
        vector = vector_type()
        assert isinstance(vector, array_type)
        assert isinstance(vector, DistributedArray[value_type])


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_distribution_and_data_access(vector_type: type[V]) -> None:
    vector = vector_type()
    assert vector.empty()
    assert vector.getSize() == 0

    begin, end = get_local_range()
    vector.setDistribution(begin, end, 2, GLOBAL_SIZE)
    assert vector.getLocalRange() == (begin, end)
    assert vector.getGhosts() == 2
    assert vector.getSize() == GLOBAL_SIZE
    assert not vector.empty()

    vector.setValue(3)
    assert_local_values(vector, 3)
    vector.setElement(begin, 4)
    assert vector[begin] == 4
    vector[begin] = 5
    assert vector.getElement(begin) == 5

    vector.reset()
    assert vector.empty()
    assert vector.getSize() == 0


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_invalid_distribution(vector_type: type[V]) -> None:
    vector = vector_type()
    begin, end = get_local_range()
    with pytest.raises(ValueError):
        vector.setDistribution(-1, end, 0, GLOBAL_SIZE)
    with pytest.raises(ValueError):
        vector.setDistribution(begin, end, -1, GLOBAL_SIZE)
    with pytest.raises(IndexError):
        vector.setDistribution(begin, GLOBAL_SIZE + 1, 0, GLOBAL_SIZE)


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_nonlocal_access(vector_type: type[V]) -> None:
    vector = create_vector(vector_type)
    begin, end = get_local_range()
    nonlocal_index = begin - 1 if begin > 0 else end

    with pytest.raises(IndexError):
        vector.getElement(nonlocal_index)
    with pytest.raises(IndexError):
        vector[nonlocal_index]
    with pytest.raises(IndexError):
        vector.setElement(nonlocal_index, 0)
    with pytest.raises(IndexError):
        vector[nonlocal_index] = 0


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_local_views(vector_type: type[V]) -> None:
    vector = create_vector(vector_type, ghosts=2)
    view = vector.getLocalView()
    const_view = vector.getConstLocalView()
    view_with_ghosts = vector.getLocalViewWithGhosts()
    const_view_with_ghosts = vector.getConstLocalViewWithGhosts()

    assert type(view) is vector_type.LocalViewType
    assert type(const_view) is vector_type.ConstLocalViewType
    assert type(view_with_ghosts) is vector_type.LocalViewType
    assert type(const_view_with_ghosts) is vector_type.ConstLocalViewType

    begin, end = get_local_range()
    local_size = end - begin
    assert view.getSize() == local_size
    assert const_view.getSize() == local_size
    assert view_with_ghosts.getSize() == local_size + 2
    assert const_view_with_ghosts.getSize() == local_size + 2

    view[0] = 42
    assert vector.getElement(begin) == 42
    view.setElement(1, 43)
    assert vector.getElement(begin + 1) == 43


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_vector_arithmetic(vector_type: type[V]) -> None:
    vector = create_vector(vector_type)
    other = vector_type()
    other.setLike(vector)
    vector.setValue(12)
    other.setValue(3)

    vector += other
    assert_local_values(vector, 15)
    vector -= other
    assert_local_values(vector, 12)
    vector *= other
    assert_local_values(vector, 36)
    vector /= other
    assert_local_values(vector, 12)


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_scalar_arithmetic(vector_type: type[V]) -> None:
    vector = create_vector(vector_type)
    vector.setValue(12)

    vector += 3
    assert_local_values(vector, 15)
    vector -= 3
    assert_local_values(vector, 12)
    vector *= 3
    assert_local_values(vector, 36)
    vector /= 3
    assert_local_values(vector, 12)


def test_modulo_int() -> None:
    vector = create_vector(pytnl._containers.DistributedVector_int)
    other = create_vector(pytnl._containers.DistributedVector_int)
    vector.setValue(17)
    other.setValue(5)

    vector %= other
    assert_local_values(vector, 2)
    vector.setValue(17)
    vector %= 4
    assert_local_values(vector, 1)


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_equality(vector_type: type[V]) -> None:
    vector = create_vector(vector_type)
    other = create_vector(vector_type)
    vector.setValue(1)
    other.setValue(1)

    assert vector == other
    vector.setValue(2)
    assert vector != other
    other.setValue(2)
    assert vector == other


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_deepcopy(vector_type: type[V]) -> None:
    vector = create_vector(vector_type, ghosts=2)
    begin, end = get_local_range()
    for i in range(begin, end):
        vector[i] = i + 1

    vector_copy = copy.deepcopy(vector)
    assert vector_copy is not vector
    assert vector_copy == vector
    for i in range(begin, end):
        assert vector_copy[i] == vector[i] == i + 1

    vector_copy[begin] = -1
    assert vector[begin] == begin + 1
    assert vector_copy != vector


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_synchronization(vector_type: type[V]) -> None:
    vector = create_vector(vector_type, ghosts=2)
    vector.setSynchronizer(1)
    vector.setValue(RANK + 1)
    vector.startSynchronization()
    vector.waitForSynchronization()

    left = (RANK - 1) % NPROC
    right = (RANK + 1) % NPROC
    begin, end = get_local_range()
    expected = [RANK + 1] * (end - begin) + [left + 1, right + 1]
    assert list(vector.getLocalViewWithGhosts()) == expected
    assert list(vector.getConstLocalViewWithGhosts()) == expected


@pytest.mark.parametrize("vector_type", VECTOR_TYPES)
def test_str(vector_type: type[V]) -> None:
    vector = create_vector(vector_type)
    assert str(vector).startswith("DistributedVector[")

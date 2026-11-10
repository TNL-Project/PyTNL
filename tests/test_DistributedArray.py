import copy
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import mpi4py
import mpi4py.MPI
import pytest

import pytnl._containers
from pytnl import devices
from pytnl._meta import VT
from pytnl.containers import Array, DistributedArray

DA = TypeVar(
    "DA",
    pytnl._containers.DistributedArray_int,
    pytnl._containers.DistributedArray_float,
    pytnl._containers.DistributedArray_complex,
)
A = TypeVar(
    "A",
    pytnl._containers.Array_int,
    pytnl._containers.Array_float,
    pytnl._containers.Array_complex,
)

pytestmark = pytest.mark.mpi

NPROC = mpi4py.MPI.COMM_WORLD.Get_size()
RANK = mpi4py.MPI.COMM_WORLD.Get_rank()
GLOBAL_SIZE = 3 * NPROC + 1

VALUE_TYPE_PARAMS = [
    (int, DistributedArray[int]),
    (float, DistributedArray[float]),
    (complex, DistributedArray[complex]),
]
ARRAY_PAIR_PARAMS = [
    (DistributedArray[int], Array[int]),
    (DistributedArray[float], Array[float]),
    (DistributedArray[complex], Array[complex]),
]


def local_range() -> tuple[int, int]:
    begin = RANK * GLOBAL_SIZE // NPROC
    end = (RANK + 1) * GLOBAL_SIZE // NPROC
    return begin, end


def make_distributed(array_type: type[DA], ghosts: int = 0) -> DA:
    array = array_type()
    begin, end = local_range()
    array.setDistribution(begin, end, ghosts, GLOBAL_SIZE)
    return array


def make_global_array(array_type: type[A]) -> A:
    array = array_type(GLOBAL_SIZE)
    for i in range(GLOBAL_SIZE):
        array[i] = i + 1
    return array


def assert_global_sequence(array: DA) -> None:
    begin, end = local_range()
    for i in range(begin, end):
        assert array.getElement(i) == i + 1


def test_pythonization() -> None:
    assert DistributedArray[int] is pytnl._containers.DistributedArray_int
    assert DistributedArray[float] is pytnl._containers.DistributedArray_float
    assert DistributedArray[complex] is pytnl._containers.DistributedArray_complex
    assert DistributedArray[int, devices.Host] is pytnl._containers.DistributedArray_int
    assert DistributedArray[float, devices.Host] is pytnl._containers.DistributedArray_float
    assert DistributedArray[complex, devices.Host] is pytnl._containers.DistributedArray_complex


def test_typedefs() -> None:
    params = (
        (DistributedArray[int], int, pytnl._containers.ArrayView_int, pytnl._containers.ArrayView_int_const),
        (DistributedArray[float], float, pytnl._containers.ArrayView_float, pytnl._containers.ArrayView_float_const),
        (DistributedArray[complex], complex, pytnl._containers.ArrayView_complex, pytnl._containers.ArrayView_complex_const),
    )
    for array_type, value_type, view_type, const_view_type in params:
        assert array_type.ValueType is value_type
        assert array_type.IndexType is int
        assert array_type.LocalViewType is view_type
        assert array_type.ConstLocalViewType is const_view_type


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_empty(value_type: type[VT], array_type: type[DA]) -> None:
    array = array_type()
    assert array.getSize() == 0
    assert array.empty()

    array = make_distributed(array_type)
    assert not array.empty()
    array.reset()
    assert array.getSize() == 0
    assert array.empty()


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_setDistribution(value_type: type[VT], array_type: type[DA]) -> None:
    array = array_type()
    begin, end = local_range()
    array.setDistribution(begin, end, 2, GLOBAL_SIZE)
    local_size = end - begin

    assert array.getSize() == GLOBAL_SIZE
    assert array.getLocalRange() == (begin, end)
    assert array.getGhosts() == 2
    assert array.getLocalView().getSize() == local_size
    assert array.getConstLocalView().getSize() == local_size
    assert array.getLocalViewWithGhosts().getSize() == local_size + 2
    assert array.getConstLocalViewWithGhosts().getSize() == local_size + 2


def test_setDistribution_errors() -> None:
    array = DistributedArray[int]()
    with pytest.raises(ValueError):
        array.setDistribution(-1, 1, 0, GLOBAL_SIZE)
    with pytest.raises(ValueError):
        array.setDistribution(1, 0, 0, GLOBAL_SIZE)
    with pytest.raises(ValueError):
        array.setDistribution(0, 1, -1, GLOBAL_SIZE)
    with pytest.raises(IndexError):
        array.setDistribution(0, GLOBAL_SIZE + 1, 0, GLOBAL_SIZE)


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_data_access(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type)
    begin, end = local_range()

    array.setValue(42)
    for i in range(begin, end):
        assert array.getElement(i) == value_type(42)
        assert array[i] == value_type(42)

    array.setElement(begin, 7)
    assert array.getElement(begin) == value_type(7)
    assert array[begin] == value_type(7)
    array[begin] = 9
    assert array.getElement(begin) == value_type(9)
    assert array[begin] == value_type(9)


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_non_local_indices(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type)
    begin, end = local_range()

    for index in (begin - 1, end):
        with pytest.raises(IndexError):
            array.getElement(index)
        with pytest.raises(IndexError):
            array.setElement(index, 0)
        with pytest.raises(IndexError):
            array[index]
        with pytest.raises(IndexError):
            array[index] = 0


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_local_view_mutation(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type, ghosts=2)
    begin, _ = local_range()
    array.setValue(0)

    local_view = array.getLocalView()
    local_view[0] = 7
    assert array.getElement(begin) == value_type(7)
    view_with_ghosts = array.getLocalViewWithGhosts()
    view_with_ghosts[0] = 8
    assert array.getElement(begin) == value_type(8)
    assert array.getConstLocalView()[0] == value_type(8)
    assert array.getConstLocalViewWithGhosts()[0] == value_type(8)


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_setLike_assign(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type, ghosts=2)
    array.setValue(RANK + 1)

    other = array_type()
    other.setLike(array)
    assert other.getSize() == GLOBAL_SIZE
    assert other.getLocalRange() == local_range()
    assert other.getGhosts() == 2

    assigned = array_type()
    assigned.assign(array)
    assert assigned == array


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
@pytest.mark.parametrize("copy_function", [copy.copy, copy.deepcopy])
def test_copy(value_type: type[VT], array_type: type[DA], copy_function: Callable[[DA], DA]) -> None:
    array = make_distributed(array_type)
    array.setValue(RANK + 1)

    constructed = array_type(array)  # pyright: ignore[reportArgumentType]
    assert constructed == array
    copied = copy_function(array)
    assert copied == array

    begin, _ = local_range()
    array[begin] = -1
    assert copied[begin] == value_type(RANK + 1)
    assert copied != array


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_equality(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type)
    other = make_distributed(array_type)
    array.setValue(0)
    other.setValue(0)
    assert array == other
    assert not array != other

    begin, _ = local_range()
    array[begin] = 1
    assert array != other
    assert not array == other


@pytest.mark.parametrize("array_type, global_array_type", ARRAY_PAIR_PARAMS)
def test_copyFromGlobal(array_type: type[DA], global_array_type: type[A]) -> None:
    array = make_distributed(array_type)
    global_array = make_global_array(global_array_type)
    array.copyFromGlobal(global_array.getConstView())  # type: ignore[arg-type]
    assert_global_sequence(array)


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_synchronization(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type, ghosts=2)
    with pytest.raises(ValueError):
        array.setSynchronizer(-1)
    with pytest.raises(ValueError):
        array.setSynchronizer(0)
    with pytest.raises(ValueError):
        array.setSynchronizer(2)
    array.setSynchronizer(1)
    array.setValue(RANK + 1)
    array.startSynchronization()
    array.waitForSynchronization()

    view = array.getLocalViewWithGhosts()
    local_size = view.getSize() - 2
    left = (RANK - 1) % NPROC
    right = (RANK + 1) % NPROC
    for i in range(local_size):
        assert view[i] == value_type(RANK + 1)
    assert view[local_size] == value_type(left + 1)
    assert view[local_size + 1] == value_type(right + 1)


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_synchronization_without_ghosts(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type)
    array.setValue(value_type(3))  # type: ignore[arg-type]
    array.setSynchronizer(0)
    array.startSynchronization()
    array.waitForSynchronization()
    begin, end = local_range()
    assert list(array.getLocalView()) == [value_type(3)] * (end - begin)


@pytest.mark.parametrize("value_type, array_type", VALUE_TYPE_PARAMS)
def test_str(value_type: type[VT], array_type: type[DA]) -> None:
    array = make_distributed(array_type, ghosts=2)
    begin, end = local_range()
    text = str(array)
    assert text.startswith("DistributedArray[")
    assert value_type.__name__ in text
    assert f"localRange=({begin}, {end})" in text


@pytest.mark.parametrize("array_type, global_array_type", ARRAY_PAIR_PARAMS)
def test_loadFromGlobalFile(array_type: type[DA], global_array_type: type[A], tmp_path: Path) -> None:
    file_name: str = mpi4py.MPI.COMM_WORLD.bcast(str(tmp_path / "global_array.tnl") if RANK == 0 else None, root=0)
    if RANK == 0:
        global_array = make_global_array(global_array_type)
        global_array.save(file_name)
    mpi4py.MPI.COMM_WORLD.Barrier()

    array = make_distributed(array_type)
    array.loadFromGlobalFile(file_name)
    mpi4py.MPI.COMM_WORLD.Barrier()
    assert_global_sequence(array)

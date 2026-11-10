# mypy: disable-error-code="import-not-found, no-any-unimported, no-untyped-call, unused-ignore"
# pyright: standard
# pyright: reportMissingImports=information

import copy
from typing import TYPE_CHECKING, Any

import mpi4py
import mpi4py.MPI
import pytest

from pytnl._meta import VT
from pytnl.containers import Array, DistributedArray
from pytnl.devices import Cuda

if TYPE_CHECKING:
    import pytnl._containers_cuda as _containers_cuda
else:
    _containers_cuda = pytest.importorskip("pytnl._containers_cuda")

# Mark all tests in this module
pytestmark = pytest.mark.cuda

NPROC = mpi4py.MPI.COMM_WORLD.Get_size()
RANK = mpi4py.MPI.COMM_WORLD.Get_rank()
GLOBAL_SIZE = 3 * NPROC + 1
VALUE_TYPES: list[type[VT]] = [int, float, complex]
LOCAL_VIEW_TYPES: dict[type[VT], type[Any]] = {
    int: _containers_cuda.ArrayView_int,
    float: _containers_cuda.ArrayView_float,
    complex: _containers_cuda.ArrayView_complex,
}
CONST_LOCAL_VIEW_TYPES: dict[type[VT], type[Any]] = {
    int: _containers_cuda.ArrayView_int_const,
    float: _containers_cuda.ArrayView_float_const,
    complex: _containers_cuda.ArrayView_complex_const,
}


def local_range() -> tuple[int, int]:
    begin = RANK * GLOBAL_SIZE // NPROC
    end = (RANK + 1) * GLOBAL_SIZE // NPROC
    return begin, end


def test_pythonization() -> None:
    assert DistributedArray[int, Cuda] is _containers_cuda.DistributedArray_int
    assert DistributedArray[float, Cuda] is _containers_cuda.DistributedArray_float
    assert DistributedArray[complex, Cuda] is _containers_cuda.DistributedArray_complex


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_typedefs(value_type: type[VT]) -> None:
    array_type = DistributedArray[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    assert array_type.ValueType is value_type
    assert array_type.IndexType is int
    assert array_type.LocalViewType is LOCAL_VIEW_TYPES[value_type]
    assert array_type.ConstLocalViewType is CONST_LOCAL_VIEW_TYPES[value_type]


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_setDistribution(value_type: type[VT]) -> None:
    begin, end = local_range()
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    assert a.getLocalRange() == (begin, end)
    assert a.getGhosts() == 0
    assert a.getSize() == GLOBAL_SIZE


@pytest.mark.parametrize("value_type", VALUE_TYPES)
@pytest.mark.parametrize("begin,end,ghosts", [(-1, 1, 0), (1, 0, 0), (0, 1, -1)])
def test_setDistribution_errors(value_type: type[VT], begin: int, end: int, ghosts: int) -> None:
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    with pytest.raises(ValueError):
        a.setDistribution(begin, end, ghosts, GLOBAL_SIZE)


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_data_access(value_type: type[VT]) -> None:
    begin, end = local_range()
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    a.setValue(42)
    for i in range(begin, end):
        assert a.getElement(i) == 42
        assert a[i] == 42
    a.setElement(begin, 7)
    assert a.getElement(begin) == 7
    assert a[begin] == 7
    a[begin] = 9
    assert a.getElement(begin) == 9
    assert a[begin] == 9


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_non_local_indices(value_type: type[VT]) -> None:
    begin, end = local_range()
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    invalid_indices = [-1, GLOBAL_SIZE]
    if NPROC > 1:
        invalid_indices.append(end % GLOBAL_SIZE)
    for i in invalid_indices:
        with pytest.raises(IndexError):
            a.getElement(i)
        with pytest.raises(IndexError):
            a[i]
        with pytest.raises(IndexError):
            a.setElement(i, 0)
        with pytest.raises(IndexError):
            a[i] = 0


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_local_views(value_type: type[VT]) -> None:
    begin, end = local_range()
    local_size = end - begin
    array_type = DistributedArray[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    a = array_type()
    a.setDistribution(begin, end, 2, GLOBAL_SIZE)
    a.setValue(5)
    local_view = a.getLocalView()
    const_local_view = a.getConstLocalView()
    view_with_ghosts = a.getLocalViewWithGhosts()
    const_view_with_ghosts = a.getConstLocalViewWithGhosts()
    assert isinstance(local_view, LOCAL_VIEW_TYPES[value_type])
    assert isinstance(const_local_view, CONST_LOCAL_VIEW_TYPES[value_type])
    assert local_view.getSize() == local_size
    assert const_local_view.getSize() == local_size
    assert view_with_ghosts.getSize() == local_size + 2
    assert const_view_with_ghosts.getSize() == local_size + 2
    for view in (local_view, const_local_view, view_with_ghosts, const_view_with_ghosts):
        for i in range(local_size):
            assert view[i] == 5
    local_view[0] = 7
    assert a[begin] == 7
    assert const_local_view[0] == 7
    assert view_with_ghosts[0] == 7
    assert const_view_with_ghosts[0] == 7
    view_with_ghosts[0] = 9
    assert a[begin] == 9
    assert local_view[0] == 9
    assert const_local_view[0] == 9


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_initialization_from_other(value_type: type[VT]) -> None:
    begin, end = local_range()
    array_type = DistributedArray[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    a = array_type()
    a.setDistribution(begin, end, 2, GLOBAL_SIZE)
    a.setValue(11)
    copy_constructed = array_type(a)
    set_like = array_type()
    set_like.setLike(a)  # pyright: ignore[reportArgumentType]
    set_like.setValue(11)
    assigned = array_type()
    assigned.assign(a)
    for other in (copy_constructed, set_like, assigned):
        assert other.getLocalRange() == (begin, end)
        assert other.getGhosts() == 2
        assert other.getSize() == GLOBAL_SIZE
        for i in range(begin, end):
            assert other[i] == 11


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_copy(value_type: type[VT]) -> None:
    begin, end = local_range()
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    a.setValue(8)
    shallow_copy = copy.copy(a)
    deep_copy = copy.deepcopy(a)
    for other in (shallow_copy, deep_copy):
        assert other.getLocalRange() == (begin, end)
        assert other.getSize() == GLOBAL_SIZE
        for i in range(begin, end):
            assert other[i] == 8
    a[begin] = 9
    assert shallow_copy[begin] == 8
    assert deep_copy[begin] == 8
    shallow_copy[begin] = 10
    assert a[begin] == 9
    assert deep_copy[begin] == 8


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_equality(value_type: type[VT]) -> None:
    begin, end = local_range()
    array_type = DistributedArray[value_type, Cuda]  # type: ignore[type-arg, valid-type]
    a = array_type()
    b = array_type()
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    b.setDistribution(begin, end, 0, GLOBAL_SIZE)
    a.setValue(42)
    b.setValue(42)
    assert a == b
    assert not a != b
    a[begin] = 1
    assert a != b
    assert not a == b
    b[begin] = 1
    assert a == b


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_copyFromGlobal(value_type: type[VT]) -> None:
    begin, end = local_range()
    global_array = Array[value_type, Cuda](GLOBAL_SIZE)  # type: ignore[type-arg, valid-type]
    for i in range(GLOBAL_SIZE):
        global_array[i] = i + 1
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    a.copyFromGlobal(global_array.getConstView())
    for i in range(begin, end):
        assert a[i] == i + 1


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_synchronization(value_type: type[VT]) -> None:
    begin, end = local_range()
    local_size = end - begin
    left = (RANK - 1) % NPROC
    right = (RANK + 1) % NPROC
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    a.setDistribution(begin, end, 2, GLOBAL_SIZE)
    a.setValue(RANK + 1)
    a.setSynchronizer(1)
    a.startSynchronization()
    a.waitForSynchronization()
    view = a.getLocalViewWithGhosts()
    const_view = a.getConstLocalViewWithGhosts()
    expected = [RANK + 1] * local_size + [left + 1, right + 1]
    assert view.getSize() == len(expected)
    assert const_view.getSize() == len(expected)
    for i, value in enumerate(expected):
        assert view[i] == value
        assert const_view[i] == value


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_empty_and_reset(value_type: type[VT]) -> None:
    begin, end = local_range()
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    assert a.empty()
    assert a.getSize() == 0
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    assert not a.empty()
    assert a.getSize() == GLOBAL_SIZE
    a.reset()
    assert a.empty()
    assert a.getSize() == 0


@pytest.mark.parametrize("value_type", VALUE_TYPES)
def test_str(value_type: type[VT]) -> None:
    begin, end = local_range()
    a = DistributedArray[value_type, Cuda]()  # type: ignore[type-arg, valid-type]
    a.setDistribution(begin, end, 0, GLOBAL_SIZE)
    expected = f"DistributedArray[{value_type.__name__}, Cuda]({GLOBAL_SIZE}, localRange=({begin}, {end}), ghosts=0)"
    assert str(a) == expected

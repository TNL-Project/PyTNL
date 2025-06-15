import copy

import pytest
from hypothesis import given
from hypothesis import strategies as st

import pytnl.containers

# ----------------------
# Configuration
# ----------------------

# List of array types to test
array_types = [
    pytnl.containers.StaticVector[1, int],
    pytnl.containers.StaticVector[2, int],
    pytnl.containers.StaticVector[3, int],
    pytnl.containers.StaticVector[1, float],
    pytnl.containers.StaticVector[2, float],
    pytnl.containers.StaticVector[3, float],
]


# ----------------------
# Helper Functions
# ----------------------


def element_strategy(array_type):
    """Return appropriate data strategy for the given array type."""
    if array_type.ValueType is int:
        # lower limits because C++ uses int64_t for IndexType
        return st.integers(min_value=-(2**63), max_value=2**63 - 1)
    else:
        return st.floats(allow_nan=False, allow_infinity=False)


def create_array(data, array_type):
    """Create an array of the given type from a list of values."""
    assert len(data) == array_type.getSize()
    return array_type(data)


# ----------------------
# Hypothesis Strategies
# ----------------------


@st.composite
def array_strategy(draw, array_type):
    """Generate an array of the given type."""
    data = draw(st.lists(element_strategy(array_type), min_size=array_type.getSize(), max_size=array_type.getSize()))
    return create_array(data, array_type)


# ----------------------
# Constructors and basic properties
# ----------------------


def test_pythonization():
    assert pytnl.containers.StaticVector[1, int] is pytnl._containers.StaticVector_1_int
    assert pytnl.containers.StaticVector[2, int] is pytnl._containers.StaticVector_2_int
    assert pytnl.containers.StaticVector[3, int] is pytnl._containers.StaticVector_3_int
    assert pytnl.containers.StaticVector[1, float] is pytnl._containers.StaticVector_1_float
    assert pytnl.containers.StaticVector[2, float] is pytnl._containers.StaticVector_2_float
    assert pytnl.containers.StaticVector[3, float] is pytnl._containers.StaticVector_3_float


def test_typedefs():
    for array_type in array_types:
        assert array_type.IndexType is int
        if array_type.__name__.endswith("_int"):
            assert array_type.ValueType is int
            assert array_type.RealType is int
        else:
            assert array_type.ValueType is float
            assert array_type.RealType is float


@pytest.mark.parametrize("array_type", array_types)
def test_constructors(array_type):
    v1 = array_type(0)
    for i in range(v1.getSize()):
        assert v1[i] == 0

    v2 = array_type(10)
    for i in range(v1.getSize()):
        assert v2[i] == 10


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_conversion(array_type, data):
    elements = data.draw(st.lists(element_strategy(array_type), min_size=array_type.getSize(), max_size=array_type.getSize()))
    # initialize StaticArray by a list
    v = array_type(elements)
    for i in range(v.getSize()):
        assert v[i] == elements[i]

    # convert the StaticArray to a list
    assert isinstance(v.as_list(), list)
    assert v.as_list() == elements
    assert list(v) == elements


# ----------------------
# Data access
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_set_get_element(array_type, data):
    elements = data.draw(st.lists(element_strategy(array_type), min_size=array_type.getSize(), max_size=array_type.getSize()))
    v = create_array(elements, array_type)
    for i in range(len(elements)):
        assert v[i] == elements[i]
        v[i] = 1
        assert v[i] == 1


@pytest.mark.parametrize("array_type", array_types)
def test_out_of_bounds_access(array_type):
    v = array_type(1)
    with pytest.raises(IndexError):
        v[-1]
    with pytest.raises(IndexError):
        v[array_type.getSize()]


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_x_property(array_type, data):
    elements = data.draw(st.lists(element_strategy(array_type), min_size=array_type.getSize(), max_size=array_type.getSize()))
    v = create_array(elements, array_type)

    assert v.x == v[0] == elements[0]
    if array_type.ValueType is int:
        new_value = -v[0] - 1
    else:
        new_value = -v[0] or 1
    v.x = new_value
    assert v.x == v[0] == new_value


@pytest.mark.parametrize("array_type", [t for t in array_types if t.getSize() >= 2])
@given(data=st.data())
def test_y_property(array_type, data):
    elements = data.draw(st.lists(element_strategy(array_type), min_size=array_type.getSize(), max_size=array_type.getSize()))
    v = create_array(elements, array_type)

    assert v.y == v[1] == elements[1]
    if array_type.ValueType is int:
        new_value = -v[1] - 1
    else:
        new_value = -v[1] or 1
    v.y = new_value
    assert v.y == v[1] == new_value


@pytest.mark.parametrize("array_type", [t for t in array_types if t.getSize() >= 3])
@given(data=st.data())
def test_z_property(array_type, data):
    elements = data.draw(st.lists(element_strategy(array_type), min_size=array_type.getSize(), max_size=array_type.getSize()))
    v = create_array(elements, array_type)

    assert v.z == v[2] == elements[2]
    if array_type.ValueType is int:
        new_value = -v[2] - 1
    else:
        new_value = -v[2] or 1
    v.z = new_value
    assert v.z == v[2] == new_value


# TODO: slicing for StaticArray is not implemented
# @pytest.mark.parametrize("array_type", array_types)
# @given(
#    data=st.data(),
#    size=st.integers(min_value=0, max_value=20),
#    start=st.integers(min_value=0, max_value=10),
#    stop=st.integers(min_value=0, max_value=20),
#    step=st.integers(min_value=1, max_value=5),
# )
# def test_slicing(array_type, data, size, start, stop, step):
#    assume(start < stop)
#    elements = data.draw(st.lists(element_strategy(array_type), min_size=size, max_size=size))
#    v = create_array(elements, array_type)
#    slice_ = slice(start, stop, step)
#    result = v[slice_]
#    expected = elements[slice_]
#    assert result.getSize() == len(expected)
#    for i in range(result.getSize()):
#        assert result[i] == expected[i]


# ----------------------
# Assignment
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_assign(array_type, data):
    v1 = data.draw(array_strategy(array_type))
    v2 = array_type(0)
    v2.assign(v1)
    assert v2.getSize() == v1.getSize()
    for i in range(v1.getSize()):
        assert v2[i] == v1[i]


# ----------------------
# Comparison operators
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_comparison_operators(array_type, data):
    size = array_type.getSize()
    elements1 = data.draw(st.lists(element_strategy(array_type), min_size=size, max_size=size))
    elements2 = data.draw(st.lists(element_strategy(array_type), min_size=size, max_size=size))
    v1 = create_array(elements1, array_type)
    v2 = create_array(elements2, array_type)

    assert (v1 == v2) == (elements1 == elements2)
    assert (v1 != v2) == (elements1 != elements2)


# ----------------------
# Fill (setValue)
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_setValue(array_type, data):
    size = array_type.getSize()
    elements = data.draw(st.lists(element_strategy(array_type), min_size=size, max_size=size))
    value = data.draw(element_strategy(array_type))

    v = create_array(elements, array_type)
    v.setValue(value)
    for i in range(size):
        assert v[i] == value


# ----------------------
# File I/O
# ----------------------


# TODO: StaticArray::save and StaticArray::load are not exposed to Python
# @pytest.mark.parametrize("array_type", array_types)
# @given(data=st.data())
# def test_save_load(array_type, data):
#    # Unfortunately functions-scoped fixtures like tmp_path do not work with Hypothesis
#    # https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
#    # Create a temporary file that will not be deleted automatically
#    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmpfile:
#        filename = tmpfile.name
#
#    try:
#        v1 = data.draw(array_strategy(array_type))
#        v1.save(str(filename))
#        v2 = array_type()
#        v2.load(str(filename))
#        assert v2.getSize() == v1.getSize()
#        for i in range(v1.getSize()):
#            assert v2[i] == v1[i]
#
#    finally:
#        # Ensure the file is deleted after the test
#        os.unlink(filename)


# ----------------------
# String representation
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
def test_str(array_type):
    v = array_type(5)
    for i in range(array_type.getSize()):
        v[i] = i
    s = str(v)
    assert isinstance(s, str)
    assert len(s) > 0


# ----------------------
# Deepcopy support
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_copy(array_type, data):
    v = data.draw(array_strategy(array_type))
    v_copy = copy.copy(v)
    assert v == v_copy
    if v.getSize() > 0:
        if array_type.ValueType is int:
            v_copy[0] = -v_copy[0] - 1
        else:
            v_copy[0] = -v_copy[0] or 1
        assert v_copy != v


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_deepcopy(array_type, data):
    v = data.draw(array_strategy(array_type))
    v_copy = copy.deepcopy(v)
    assert v == v_copy
    if v.getSize() > 0:
        if array_type.ValueType is int:
            v_copy[0] = -v_copy[0] - 1
        else:
            v_copy[0] = -v_copy[0] or 1
        assert v_copy != v

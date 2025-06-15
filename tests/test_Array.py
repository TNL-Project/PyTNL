import copy
import os
import tempfile

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import pytnl.containers

# ----------------------
# Configuration
# ----------------------

# List of array types to test
array_types = [
    pytnl.containers.Array[int],
    pytnl.containers.Array[float],
    pytnl.containers.Vector[int],
    pytnl.containers.Vector[float],
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
    v = array_type(len(data))
    for i, val in enumerate(data):
        v[i] = val
    return v


# ----------------------
# Hypothesis Strategies
# ----------------------


@st.composite
def array_strategy(draw, array_type):
    """Generate an array of the given type."""
    data = draw(st.lists(element_strategy(array_type), max_size=20))
    return create_array(data, array_type)


# ----------------------
# Constructors and basic properties
# ----------------------


def test_pythonization():
    assert pytnl.containers.Array[bool] is pytnl._containers.Array_bool
    assert pytnl.containers.Array[int] is pytnl._containers.Array_int
    assert pytnl.containers.Array[float] is pytnl._containers.Array_float
    assert pytnl.containers.Vector[int] is pytnl._containers.Vector_int
    assert pytnl.containers.Vector[float] is pytnl._containers.Vector_float


def test_typedefs():
    for array_type in array_types:
        assert array_type.IndexType is int

    assert pytnl.containers.Array[bool].ValueType is bool
    assert pytnl.containers.Array[int].ValueType is int
    assert pytnl.containers.Array[float].ValueType is float

    assert pytnl.containers.Vector[float].ValueType is float
    assert pytnl.containers.Vector[int].ValueType is int


@pytest.mark.parametrize("array_type", array_types)
def test_constructors(array_type):
    v1 = array_type()
    assert v1.getSize() == 0

    v2 = array_type(10)
    assert v2.getSize() == 10

    value = 3.14 if array_type.ValueType is float else 3
    v3 = array_type(5, value)
    assert v3.getSize() == 5
    for i in range(5):
        assert v3[i] == value

    with pytest.raises(ValueError):
        array_type(-1)


# ----------------------
# Size management
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(size=st.integers(min_value=0, max_value=20))
def test_setSize(array_type, size):
    v = array_type()
    v.setSize(size)
    assert v.getSize() == size


@pytest.mark.parametrize("array_type", array_types)
@given(size=st.integers(min_value=-20, max_value=-1))
def test_setSize_negative(array_type, size):
    v = array_type()
    with pytest.raises(ValueError):
        v.setSize(size)


@pytest.mark.parametrize("array_type", array_types)
@given(size=st.integers(min_value=0, max_value=20))
def test_resize(array_type, size):
    v = array_type()
    v.resize(size)
    assert v.getSize() == size


@pytest.mark.parametrize("array_type", array_types)
@given(size=st.integers(min_value=-20, max_value=-1))
def test_resize_negative(array_type, size):
    v = array_type()
    with pytest.raises(ValueError):
        v.resize(size)


@pytest.mark.parametrize("array_type", array_types)
@given(size=st.integers(min_value=0, max_value=20), value=st.integers(min_value=-(2**63), max_value=2**63 - 1))
def test_resize_with_value(array_type, size, value):
    if array_type.ValueType is float:
        value = float(value)
    v = array_type()
    v.resize(size, value)
    assert v.getSize() == size
    for i in range(size):
        assert v[i] == value


@pytest.mark.parametrize("array_type", array_types)
def test_swap(array_type):
    v1 = array_type(5)
    v2 = array_type(10)
    v1.swap(v2)
    assert v1.getSize() == 10
    assert v2.getSize() == 5


@pytest.mark.parametrize("array_type", array_types)
def test_reset(array_type):
    v = array_type(10)
    v.reset()
    assert v.getSize() == 0


@pytest.mark.parametrize("array_type", array_types)
def test_empty(array_type):
    v = array_type()
    assert v.empty()
    v.setSize(1)
    assert not v.empty()


# ----------------------
# Data access
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_set_get_element(array_type, data):
    elements = data.draw(st.lists(element_strategy(array_type), min_size=0, max_size=20))
    v = create_array(elements, array_type)
    for i in range(len(elements)):
        assert v[i] == elements[i]
        assert v.getElement(i) == elements[i]
        v.setElement(i, 42)
        assert v[i] == 42
        assert v.getElement(i) == 42
        v[i] = 1
        assert v[i] == 1
        assert v.getElement(i) == 1


@pytest.mark.parametrize("array_type", array_types)
def test_out_of_bounds_access(array_type):
    v = array_type(1)
    with pytest.raises(IndexError):
        v[-1]
    with pytest.raises(IndexError):
        v.getElement(-1)
    with pytest.raises(IndexError):
        v.setElement(-1, 0)
    with pytest.raises(IndexError):
        v[1]
    with pytest.raises(IndexError):
        v.getElement(1)
    with pytest.raises(IndexError):
        v.setElement(1, 0)


@pytest.mark.parametrize("array_type", array_types)
@given(
    data=st.data(),
    size=st.integers(min_value=0, max_value=20),
    start=st.integers(min_value=0, max_value=10),
    stop=st.integers(min_value=0, max_value=20),
    step=st.integers(min_value=1, max_value=5),
)
def test_slicing(array_type, data, size, start, stop, step):
    assume(start < stop)
    elements = data.draw(st.lists(element_strategy(array_type), min_size=size, max_size=size))
    v = create_array(elements, array_type)
    slice_ = slice(start, stop, step)
    result = v[slice_]
    expected = elements[slice_]
    assert result.getSize() == len(expected)
    for i in range(result.getSize()):
        assert result[i] == expected[i]


# ----------------------
# Assignment
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_assign(array_type, data):
    v1 = data.draw(array_strategy(array_type))
    v2 = array_type()
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
    size = data.draw(st.integers(min_value=0, max_value=20))
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
@given(
    data=st.data(),
    size=st.integers(min_value=0, max_value=20),
    begin=st.integers(min_value=0, max_value=20),
    end=st.integers(min_value=0, max_value=20),
)
def test_setValue(array_type, data, size, begin, end):
    assume(begin <= end <= size)
    elements = data.draw(st.lists(element_strategy(array_type), min_size=size, max_size=size))
    value = data.draw(element_strategy(array_type))

    v = create_array(elements, array_type)
    v.setValue(value, begin, end)
    # adjust according to C++ behavior
    if end == 0:
        end = size
    for i in range(size):
        if begin <= i < end:
            assert v[i] == value
        else:
            assert v[i] == elements[i]


# ----------------------
# File I/O
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
def test_serialization_type(array_type):
    assert array_type.getSerializationType().startswith("TNL::Containers::Array<")


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_save_load(array_type, data):
    # Unfortunately functions-scoped fixtures like tmp_path do not work with Hypothesis
    # https://hypothesis.readthedocs.io/en/latest/reference/api.html#hypothesis.HealthCheck.function_scoped_fixture
    # Create a temporary file that will not be deleted automatically
    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmpfile:
        filename = tmpfile.name

    try:
        v1 = data.draw(array_strategy(array_type))
        v1.save(str(filename))
        v2 = array_type()
        v2.load(str(filename))
        assert v2.getSize() == v1.getSize()
        for i in range(v1.getSize()):
            assert v2[i] == v1[i]

    finally:
        # Ensure the file is deleted after the test
        os.unlink(filename)


# ----------------------
# String representation
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
def test_str(array_type):
    v = array_type(5)
    for i in range(5):
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


# ----------------------
# Buffer protocol (Numpy interoperability)
# ----------------------


@pytest.mark.parametrize("array_type", array_types)
@given(data=st.data())
def test_buffer_protocol(array_type, data):
    tnl_a = data.draw(array_strategy(array_type))
    np_a = np.array(tnl_a)
    assert isinstance(np_a, np.ndarray)
    assert np_a.shape == (tnl_a.getSize(),)
    for i in range(tnl_a.getSize()):
        assert np_a[i] == tnl_a[i]

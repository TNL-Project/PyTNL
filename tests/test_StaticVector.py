import copy
import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import tnl

# ----------------------
# Configuration
# ----------------------

# List of vector types to test
vector_types = [
    tnl.Grid1D.CoordinatesType,
    tnl.Grid2D.CoordinatesType,
    tnl.Grid3D.CoordinatesType,
    tnl.Grid1D.PointType,
    tnl.Grid2D.PointType,
    tnl.Grid3D.PointType,
]


# ----------------------
# Helper Functions
# ----------------------


def element_strategy(vector_type):
    """Return appropriate data strategy for the given vector type."""
    if vector_type.ValueType is int:
        # lower limits because C++ uses int64_t for IndexType and we test even multiplication
        return st.integers(min_value=-(2**31), max_value=2**31)
    else:
        return st.floats(allow_nan=False, allow_infinity=False)


def create_vector(data, vector_type):
    """Create a vector of the given type from a list of values."""
    assert len(data) == vector_type.getSize()
    return vector_type(data)


# ----------------------
# Hypothesis Strategies
# ----------------------


@st.composite
def vector_strategy(draw, vector_type):
    """Generate a vector of the given type."""
    data = draw(st.lists(element_strategy(vector_type), min_size=vector_type.getSize(), max_size=vector_type.getSize()))
    return create_vector(data, vector_type)


@st.composite
def vector_pair_strategy(draw, vector_type):
    """Generate two vectors of the same size and type."""
    return draw(vector_strategy(vector_type)), draw(vector_strategy(vector_type))


@st.composite
def vector_scalar_strategy(draw, vector_type):
    """Generate a vector and a scalar of the same type."""
    v = draw(vector_strategy(vector_type))
    s = draw(element_strategy(vector_type))
    return v, s


# ----------------------
# Unary Operators
# ----------------------


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_unary_plus(vector_type, data):
    v = data.draw(vector_strategy(vector_type))
    v2 = +v
    for i in range(v.getSize()):
        assert v2[i] == +v[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_unary_minus(vector_type, data):
    v = data.draw(vector_strategy(vector_type))
    v2 = -v
    for i in range(v.getSize()):
        assert v2[i] == -v[i]


# ----------------------
# Vector OP Vector
# ----------------------


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_plus_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    v3 = v1 + v2
    for i in range(v1.getSize()):
        assert v3[i] == v1[i] + v2[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_minus_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    v3 = v1 - v2
    for i in range(v1.getSize()):
        assert v3[i] == v1[i] - v2[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_mult_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    v3 = v1 * v2
    for i in range(v1.getSize()):
        assert v3[i] == v1[i] * v2[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def teset_vector_div_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    for i in range(v2.getSize()):
        if v2[i] == 0:
            assume(False)
    v3 = v1 / v2
    for i in range(v1.getSize()):
        if vector_type.RealType is int:
            # GOTCHA: Python rounds to the next lower integer (floor), whereas C++ rounds towards zero (trunc)
            assert v3[i] == math.trunc(v1[i] / v2[i])
        else:
            assert v3[i] == v1[i] / v2[i]


# ----------------------
# Vector OP Scalar
# ----------------------


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_plus_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    v2 = v + s
    for i in range(v.getSize()):
        assert v2[i] == v[i] + s


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_minus_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    v2 = v - s
    for i in range(v.getSize()):
        assert v2[i] == v[i] - s


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_mult_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    v2 = v * s
    for i in range(v.getSize()):
        assert v2[i] == v[i] * s


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_div_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    if s == 0:
        assume(False)
    v2 = v / s
    for i in range(v.getSize()):
        if vector_type.RealType is int:
            # GOTCHA: Python rounds to the next lower integer (floor), whereas C++ rounds towards zero (trunc)
            assert v2[i] == math.trunc(v[i] / s)
        else:
            assert v2[i] == v[i] / s


# ----------------------
# Scalar OP Vector
# ----------------------


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_scalar_plus_vector(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    v2 = s + v
    for i in range(v.getSize()):
        assert v2[i] == s + v[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_scalar_minus_vector(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    v2 = s - v
    for i in range(v.getSize()):
        assert v2[i] == s - v[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_scalar_mult_vector(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    v2 = s * v
    for i in range(v.getSize()):
        assert v2[i] == s * v[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_scalar_div_vector(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    for i in range(v.getSize()):
        if v[i] == 0:
            assume(False)
    v2 = s / v
    for i in range(v.getSize()):
        if vector_type.RealType is int:
            # GOTCHA: Python rounds to the next lower integer (floor), whereas C++ rounds towards zero (trunc)
            assert v2[i] == math.trunc(s / v[i])
        else:
            assert v2[i] == s / v[i]


# ----------------------
# Modulo Tests (Only for Vector_int)
# ----------------------


@pytest.mark.parametrize("vector_type", [t for t in vector_types if t.ValueType is int])
@given(data=st.data())
def test_vector_modulo_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    for i in range(v2.getSize()):
        if v2[i] == 0:
            assume(False)
    v3 = v1 % v2
    for i in range(v1.getSize()):
        # GOTCHA: Python uses the sign of the divisor, whereas C++ uses the sign of the dividend
        # assert v3[i] == v1[i] % v2[i]
        assert v3[i] == int(math.fmod(v1[i], v2[i]))


@pytest.mark.parametrize("vector_type", [t for t in vector_types if t.ValueType is int])
@given(data=st.data())
def test_vector_modulo_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    if s == 0:
        assume(False)
    v2 = v % s
    for i in range(v.getSize()):
        # GOTCHA: Python uses the sign of the divisor, whereas C++ uses the sign of the dividend
        # assert v2[i] == v[i] % s
        assert v2[i] == int(math.fmod(v[i], s))


@pytest.mark.parametrize("vector_type", [t for t in vector_types if t.ValueType is int])
@given(data=st.data())
def test_scalar_modulo_vector(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    for i in range(v.getSize()):
        if v[i] == 0:
            assume(False)
    v2 = s % v
    for i in range(v.getSize()):
        # GOTCHA: Python uses the sign of the divisor, whereas C++ uses the sign of the dividend
        # assert v2[i] == s % v[i]
        assert v2[i] == int(math.fmod(s, v[i]))


# ----------------------
# In-place Vector OP= Vector
# ----------------------


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_iadd_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    original_v1 = copy.deepcopy(v1)
    v1 += v2
    for i in range(v1.getSize()):
        assert v1[i] == original_v1[i] + v2[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_isub_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    original_v1 = copy.deepcopy(v1)
    v1 -= v2
    for i in range(v1.getSize()):
        assert v1[i] == original_v1[i] - v2[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_imul_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    original_v1 = copy.deepcopy(v1)
    v1 *= v2
    for i in range(v1.getSize()):
        assert v1[i] == original_v1[i] * v2[i]


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_idiv_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    for i in range(v2.getSize()):
        if v2[i] == 0:
            assume(False)
    original_v1 = copy.deepcopy(v1)
    v1 /= v2
    for i in range(v1.getSize()):
        if vector_type.RealType is int:
            # GOTCHA: Python rounds to the next lower integer (floor), whereas C++ rounds towards zero (trunc)
            assert v1[i] == math.trunc(original_v1[i] / v2[i])
        else:
            assert v1[i] == original_v1[i] / v2[i]


@pytest.mark.parametrize("vector_type", [t for t in vector_types if t.ValueType is int])
@given(data=st.data())
def test_imod_vector(vector_type, data):
    v1, v2 = data.draw(vector_pair_strategy(vector_type))
    for i in range(v2.getSize()):
        if v2[i] == 0:
            assume(False)
    original_v1 = copy.deepcopy(v1)
    v1 %= v2
    for i in range(v1.getSize()):
        # GOTCHA: Python uses the sign of the divisor, whereas C++ uses the sign of the dividend
        # assert v3[i] == v1[i] % v2[i]
        assert v1[i] == int(math.fmod(original_v1[i], v2[i]))


# ----------------------
# In-place Vector OP= Scalar
# ----------------------


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_iadd_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    original_v = copy.deepcopy(v)
    v += s
    for i in range(v.getSize()):
        assert v[i] == original_v[i] + s


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_isub_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    original_v = copy.deepcopy(v)
    v -= s
    for i in range(v.getSize()):
        assert v[i] == original_v[i] - s


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_imul_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    original_v = copy.deepcopy(v)
    v *= s
    for i in range(v.getSize()):
        assert v[i] == original_v[i] * s


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_idiv_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    if s == 0:
        assume(False)
    original_v = copy.deepcopy(v)
    v /= s
    for i in range(v.getSize()):
        if vector_type.RealType is int:
            # GOTCHA: Python rounds to the next lower integer (floor), whereas C++ rounds towards zero (trunc)
            assert v[i] == math.trunc(original_v[i] / s)
        else:
            assert v[i] == original_v[i] / s


@pytest.mark.parametrize("vector_type", [t for t in vector_types if t.ValueType is int])
@given(data=st.data())
def test_imod_scalar(vector_type, data):
    v, s = data.draw(vector_scalar_strategy(vector_type))
    if s == 0:
        assume(False)
    original_v = copy.deepcopy(v)
    v %= s
    for i in range(v.getSize()):
        # GOTCHA: Python uses the sign of the divisor, whereas C++ uses the sign of the dividend
        # assert v3[i] == v1[i] % v2[i]
        assert v[i] == int(math.fmod(original_v[i], s))


# ----------------------
# Comparison operators
# ----------------------


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_eq(vector_type, data):
    size = vector_type.getSize()
    elements1 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    elements2 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    v1 = create_vector(elements1, vector_type)
    v2 = create_vector(elements2, vector_type)
    assert (v1 == v2) == (elements1 == elements2)


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_ne(vector_type, data):
    size = vector_type.getSize()
    elements1 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    elements2 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    v1 = create_vector(elements1, vector_type)
    v2 = create_vector(elements2, vector_type)
    assert (v1 != v2) == (elements1 != elements2)


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_lt(vector_type, data):
    size = vector_type.getSize()
    elements1 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    elements2 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    v1 = create_vector(elements1, vector_type)
    v2 = create_vector(elements2, vector_type)
    assert (v1 < v2) == (elements1 < elements2)


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_le(vector_type, data):
    size = vector_type.getSize()
    elements1 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    elements2 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    v1 = create_vector(elements1, vector_type)
    v2 = create_vector(elements2, vector_type)
    assert (v1 <= v2) == (elements1 <= elements2)


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_gt(vector_type, data):
    size = vector_type.getSize()
    elements1 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    elements2 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    v1 = create_vector(elements1, vector_type)
    v2 = create_vector(elements2, vector_type)
    assert (v1 > v2) == (elements1 > elements2)


@pytest.mark.parametrize("vector_type", vector_types)
@given(data=st.data())
def test_vector_ge(vector_type, data):
    size = vector_type.getSize()
    elements1 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    elements2 = data.draw(st.lists(element_strategy(vector_type), min_size=size, max_size=size))
    v1 = create_vector(elements1, vector_type)
    v2 = create_vector(elements2, vector_type)
    assert (v1 >= v2) == (elements1 >= elements2)

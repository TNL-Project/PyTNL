import pytest

import tnl


@pytest.mark.parametrize("size", [0, 1, 10, 100, 1000, 2**30])
def test_init(size):
    v = tnl.Vector(size)
    assert v.getSize() == size


def test_square_brackets(size):
    v = tnl.Vector(size)
    for i in range(size):
        v[i] = i
    for i in range(size):
        assert v[i] == i


def test_addition(size):
    v1 = v2 = tnl.Vector(size)
    for i in range(size):
        v1[i] = i
        v2[i] = i * 2
    v3 = v1 + v2
    for i in range(size):
        assert v3[i] == v1[i] + v2[i]

def test_negative_size():
    with pytest.raises(ValueError):
        tnl.Vector(-1)

def test_copy(size):
    v1 = tnl.Vector(size)
    for i in range(size):
        v1[i] = i
    v2 = v1.copy()
    assert v1 == v2
    v2[0] = 99
    assert v1 != v2

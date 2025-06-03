import pytest

import tnl


def test_setElement() -> None:
    a = tnl.Array()
    with pytest.raises(IndexError):
        a.setElement(0, 1)


def test_getElement() -> None:
    a = tnl.Array()
    with pytest.raises(IndexError):
        a.getElement(0)

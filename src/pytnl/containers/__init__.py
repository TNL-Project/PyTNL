from typing import Any, Literal, overload

import pytnl._containers
import pytnl._meta
from pytnl._meta import DIMS, VT

__all__ = [
    "Array",
    "NDArray",
    "NDArrayIndexer",
    "StaticVector",
    "Vector",
]


class _ArrayMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._containers
    _class_prefix = "Array"
    _template_parameters = (("value_type", type),)

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload
    def __getitem__(self, key: type[bool]) -> type[pytnl._containers.Array_bool]: ...  # type: ignore[overload-overlap]

    @overload
    def __getitem__(self, key: type[int]) -> type[pytnl._containers.Array_int]: ...

    @overload
    def __getitem__(self, key: type[float]) -> type[pytnl._containers.Array_float]: ...

    @overload
    def __getitem__(self, key: type[complex]) -> type[pytnl._containers.Array_complex]: ...

    def __getitem__(self, key: type[bool | VT]) -> type[Any]:
        items = (key,)
        return self._get_cpp_class(items)


class Array(metaclass=_ArrayMeta):
    """
    Allows `Array[value_type]` syntax to resolve to the appropriate C++ `Array` class.

    This class provides a Python interface to C++ arrays of a specific value type.

    Examples:
    - `Array[int]` → `Array_int`
    - `Array[float]` → `Array_float`
    - `Array[complex]` → `Array_complex`
    """


class _VectorMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._containers
    _class_prefix = "Vector"
    _template_parameters = (("value_type", type),)

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload
    def __getitem__(self, key: type[int]) -> type[pytnl._containers.Vector_int]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __getitem__(self, key: type[float]) -> type[pytnl._containers.Vector_float]: ...

    @overload
    def __getitem__(self, key: type[complex]) -> type[pytnl._containers.Vector_complex]: ...

    def __getitem__(self, key: type[VT]) -> type[Any]:
        items = (key,)
        return self._get_cpp_class(items)


class Vector(metaclass=_VectorMeta):
    """
    Allows `Vector[value_type]` syntax to resolve to the appropriate C++ `Vector` class.

    This class provides a Python interface to C++ vectors of a specific value type.

    Examples:
    - `Vector[int]` → `Vector_int`
    - `Vector[float] → `Vector_float`
    - `Vector[complex] → `Vector_complex`
    """


class _StaticVectorMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._containers
    _class_prefix = "StaticVector"
    _template_parameters = (
        ("dimension", int),
        ("value_type", type),
    )

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload
    def __getitem__(self, key: tuple[Literal[1], type[int]]) -> type[pytnl._containers.StaticVector_1_int]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __getitem__(self, key: tuple[Literal[2], type[int]]) -> type[pytnl._containers.StaticVector_2_int]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __getitem__(self, key: tuple[Literal[3], type[int]]) -> type[pytnl._containers.StaticVector_3_int]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __getitem__(self, key: tuple[Literal[1], type[float]]) -> type[pytnl._containers.StaticVector_1_float]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[2], type[float]]) -> type[pytnl._containers.StaticVector_2_float]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[3], type[float]]) -> type[pytnl._containers.StaticVector_3_float]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[1], type[complex]]) -> type[pytnl._containers.StaticVector_1_complex]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[2], type[complex]]) -> type[pytnl._containers.StaticVector_2_complex]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[3], type[complex]]) -> type[pytnl._containers.StaticVector_3_complex]: ...

    def __getitem__(self, key: tuple[DIMS, type[VT]]) -> type[Any]:
        return self._get_cpp_class(key)


class StaticVector(metaclass=_StaticVectorMeta):
    """
    Allows `StaticVector[dimension, value_type]` syntax to resolve to the appropriate C++ `StaticVector` class.

    This class provides a Python interface to C++ static vectors with a fixed dimension and value type.

    Examples:
    - `StaticVector[3, float]` → `StaticVector_3_float`
    - `StaticVector[2, int]` → `StaticVector_2_int`
    """


class _NDArrayMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._containers
    _class_prefix = "NDArray"
    _template_parameters = (
        ("dimension", int),
        ("value_type", type),
    )

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload
    def __getitem__(self, key: tuple[Literal[1], type[int]]) -> type[pytnl._containers.NDArray_1_int]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __getitem__(self, key: tuple[Literal[2], type[int]]) -> type[pytnl._containers.NDArray_2_int]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __getitem__(self, key: tuple[Literal[3], type[int]]) -> type[pytnl._containers.NDArray_3_int]: ...  # pyright: ignore[reportOverlappingOverload]

    @overload
    def __getitem__(self, key: tuple[Literal[1], type[float]]) -> type[pytnl._containers.NDArray_1_float]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[2], type[float]]) -> type[pytnl._containers.NDArray_2_float]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[3], type[float]]) -> type[pytnl._containers.NDArray_3_float]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[1], type[complex]]) -> type[pytnl._containers.NDArray_1_complex]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[2], type[complex]]) -> type[pytnl._containers.NDArray_2_complex]: ...

    @overload
    def __getitem__(self, key: tuple[Literal[3], type[complex]]) -> type[pytnl._containers.NDArray_3_complex]: ...

    def __getitem__(self, key: tuple[DIMS, type[VT]]) -> type[Any]:
        return self._get_cpp_class(key)


class NDArray(metaclass=_NDArrayMeta):
    """
    Allows `NDArray[dimension, value_type]` syntax to resolve to the appropriate C++ `NDArray` class.

    This class provides a Python interface to C++ N-dimensional arrays with a fixed dimension and value type.

    Examples:
    - `NDArray[3, float]` → `NDArray_3_float`
    - `NDArray[2, int]` → `NDArray_2_int`
    """


class _NDArrayIndexerMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._containers
    _class_prefix = "NDArrayIndexer"
    _template_parameters = (("dimension", int),)

    @overload
    def __getitem__(self, key: Literal[1]) -> type[pytnl._containers.NDArrayIndexer_1]: ...

    @overload
    def __getitem__(self, key: Literal[2]) -> type[pytnl._containers.NDArrayIndexer_2]: ...

    @overload
    def __getitem__(self, key: Literal[3]) -> type[pytnl._containers.NDArrayIndexer_3]: ...

    def __getitem__(self, key: DIMS) -> type[Any]:
        items = (key,)
        return self._get_cpp_class(items)


class NDArrayIndexer(metaclass=_NDArrayIndexerMeta):
    """
    Allows `NDArrayIndexer[dimension]` syntax to resolve to the appropriate C++ `NDArrayIndexer` class.

    This class provides a Python interface to C++ indexers for N-dimensional arrays with a fixed dimension.

    Examples:
    - `NDArrayIndexer[1]` → `NDArrayIndexer_1`
    - `NDArrayIndexer[2]` → `NDArrayIndexer_2`
    """

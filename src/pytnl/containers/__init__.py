import pytnl._containers


class Array:
    @classmethod
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"{cls} should not be instantiated: use {cls.__name__}[...]")

    @classmethod
    def __class_getitem__(cls, items):
        """
        Allows Array[value_type] syntax to resolve to the correct C++ exported class.

        Example:
            Array[float] → Array_float
            Array[int]   → Array_int
        """
        if not isinstance(items, tuple):
            items = (items,)
        if len(items) != 1:
            raise TypeError("Array must be subscripted with one argument: value type")

        value_type = items[0]

        if not isinstance(value_type, type):
            raise TypeError(f"Value type must be a type, got {value_type}")

        class_name = f"Array_{value_type.__name__}"

        if not hasattr(pytnl._containers, class_name):
            raise ValueError(f"Class '{class_name}' not found in module. Ensure it is properly exported from C++.")

        return getattr(pytnl._containers, class_name)


class Vector:
    @classmethod
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"{cls} should not be instantiated: use {cls.__name__}[...]")

    @classmethod
    def __class_getitem__(cls, items):
        """
        Allows Vector[value_type] syntax to resolve to the correct C++ exported class.

        Example:
            Vector[float] → Vector_float
            Vector[int]   → Vector_int
        """
        if not isinstance(items, tuple):
            items = (items,)
        if len(items) != 1:
            raise TypeError("Vector must be subscripted with one argument: value type")

        value_type = items[0]

        if not isinstance(value_type, type):
            raise TypeError(f"Value type must be a type, got {value_type}")

        class_name = f"Vector_{value_type.__name__}"

        if not hasattr(pytnl._containers, class_name):
            raise ValueError(f"Class '{class_name}' not found in module. Ensure it is properly exported from C++.")

        return getattr(pytnl._containers, class_name)


class StaticVector:
    @classmethod
    def __new__(cls, *args, **kwargs):
        raise RuntimeError(f"{cls} should not be instantiated: use {cls.__name__}[...]")

    @classmethod
    def __class_getitem__(cls, items):
        """
        Allows StaticVector[dim, value_type] syntax to resolve to the correct C++ exported class.

        Example:
            StaticVector[3, float] → StaticVector_3_float
            StaticVector[2, int]   → StaticVector_2_int
        """
        if not isinstance(items, tuple):
            items = (items,)
        if not isinstance(items, tuple) or len(items) != 2:
            raise TypeError("StaticVector must be subscripted with two arguments: (dimension, value_type)")

        dimension, value_type = items

        if not isinstance(dimension, int):
            raise TypeError(f"Dimension must be an int, got {dimension}")
        if not isinstance(value_type, type):
            raise TypeError(f"Value type must be a type, got {value_type}")

        class_name = f"StaticVector_{dimension}_{value_type.__name__}"

        if not hasattr(pytnl._containers, class_name):
            raise ValueError(f"Class '{class_name}' not found in module. Ensure it is properly exported from C++.")

        return getattr(pytnl._containers, class_name)

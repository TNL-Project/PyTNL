from types import ModuleType
from typing import Any, Literal, TypeGuard, cast, get_args

# value types
type VT = int | float | complex

# static dimensions
type DIMS = Literal[1, 2, 3]

# general type for the `items` argument in `__getitem__`
type ItemType = type[Any] | DIMS


def is_dim_guard(dim: int) -> TypeGuard[DIMS]:
    """Verify if given `dim` satisfies the `DIMS` literal type at runtime."""
    DimsType = DIMS.__value__
    return dim in get_args(DimsType)


# Unfortunately __class_getitem__ is reserved for typing generics.
# But using __getitem__ in a metaclass allows us to implement the same API
# with non-reserved elements.
# See https://stackoverflow.com/a/77754379
class CPPClassTemplate(type):
    """
    Base metaclass for C++ class template wrappers.

    This class provides shared functionality for dynamically resolving C++ classes
    based on type parameters using Python's `__class_getitem__` syntax.
    Prevents direct instantiation and allows the target module and class lookup
    to be configured.

    Class attributes:

        _cpp_module (ModuleType):
            The Python module where C++ classes are looked up.

        _class_prefix (str):
            The prefix of the C++ classes to look up.

        _template_parameters (tuple[str, type]):
            The arguments of the `__class_getitem__` method to validate.
    """

    # Configurable module for C++ class resolution
    _cpp_module: ModuleType

    # Configurable prefix of the C++ class
    _class_prefix: str

    # Configurable tuple of arguments for the __getitem__ method
    _template_parameters: tuple[tuple[str, type], ...]

    def __getitem__(self, items: ItemType | tuple[ItemType, ...]) -> type[Any]:
        """
        Resolves the appropriate C++-exported class based on the provided type parameters.

        Validates the input and constructs the class name using `_validate_params`.
        Raises an error if the class is not found in the configured module.
        """
        if not isinstance(items, tuple):
            items = (items,)

        module = self._cpp_module
        class_name = CPPClassTemplate._validate_params(self, items)

        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_name}' not found in module '{module.__name__}'. Ensure it is properly exported from C++.")
        return cast(type[Any], getattr(module, class_name))

    def _validate_params(self, items: tuple[ItemType, ...]) -> str:
        """
        Validates the parameters passed to `__class_getitem__` and returns the corresponding class name.
        """
        item_names = [name for name, _ in self._template_parameters]
        if len(items) != len(item_names):
            raise TypeError(f"{self.__name__} must be subscripted with {len(item_names)} arguments: {item_names}")

        class_name = self._class_prefix

        for item, pair in zip(items, self._template_parameters):
            if pair[1] is type:
                if not isinstance(item, type):
                    raise TypeError(f"{pair[0]} must be a type, got {item}")
                class_name += f"_{item.__name__}"
            else:
                if not isinstance(item, pair[1]):
                    raise TypeError(f"{pair[0]} must be an instance of {pair[1].__name__}, got {item}")
                class_name += f"_{item}"

        return class_name

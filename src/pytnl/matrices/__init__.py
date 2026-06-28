from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pytnl._matrices
import pytnl._meta
import pytnl.devices
from pytnl._matrices import formats

if TYPE_CHECKING:
    # This is an optional module - at runtime it is lazy-imported in
    # `CPPClassTemplate`, for type checking there must be the import statement.
    import pytnl._matrices_cuda as _matrices_cuda  # type: ignore[import-not-found, unused-ignore]

__all__ = [
    "DenseMatrix",
    "DenseMatrixConstRowView",
    "DenseMatrixRowView",
    "DenseMatrixView",
    "SparseMatrix",
    "SparseMatrixConstRowView",
    "SparseMatrixRowView",
    "SparseMatrixView",
    "copySparseMatrix",
    "formats",
]


class _SparseMatrixMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "SparseMatrix"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
        ("format", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(self, key: type[float], /) -> type[pytnl._matrices.SparseMatrix_float_CSR]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_float_CSR]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_float_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.CSR]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_float_CSR]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.Ellpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_float_Ellpack]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.SlicedEllpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_float_SlicedEllpack]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], type[formats.CSR]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_float_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], type[formats.Ellpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_float_Ellpack]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], type[formats.SlicedEllpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_float_SlicedEllpack]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[type[Any], type[Any]] | tuple[type[Any], type[Any], type[Any]],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            # SparseMatrix[float] → default device=Host, format=CSR
            items: tuple[Any, ...] = (key, pytnl.devices.Host, formats.CSR)
        elif len(key) == 2:
            # SparseMatrix[float, Host] → device=given, format=CSR
            items = (key[0], key[1], formats.CSR)
        else:
            # SparseMatrix[float, Host, formats.CSR]
            items = key

        value_type = items[0]
        if value_type is not float:
            raise TypeError(f"SparseMatrix supports only float value type, got {value_type.__name__}")
        return self._get_cpp_class(items)


class _SparseMatrixViewMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "SparseMatrixView"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
        ("format", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(self, key: type[float], /) -> type[pytnl._matrices.SparseMatrixView_float_CSR]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_float_CSR]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_float_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.CSR]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_float_CSR]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.Ellpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_float_Ellpack]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.SlicedEllpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_float_SlicedEllpack]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], type[formats.CSR]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_float_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], type[formats.Ellpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_float_Ellpack]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], type[formats.SlicedEllpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_float_SlicedEllpack]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[type[Any], type[Any]] | tuple[type[Any], type[Any], type[Any]],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            # SparseMatrixView[float] → default device=Host, format=CSR
            items: tuple[Any, ...] = (key, pytnl.devices.Host, formats.CSR)
        elif len(key) == 2:
            # SparseMatrixView[float, Host] → device=given, format=CSR
            items = (key[0], key[1], formats.CSR)
        else:
            # SparseMatrixView[float, Host, formats.CSR]
            items = key

        value_type = items[0]
        if value_type is not float:
            raise TypeError(f"SparseMatrixView supports only float value type, got {value_type.__name__}")
        return self._get_cpp_class(items)


class _DenseMatrixMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "DenseMatrix"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(self, key: type[float], /) -> type[pytnl._matrices.DenseMatrix_float]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_float]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrix_float]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[type[Any], type[Any]],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            # DenseMatrix[float] → default device=Host
            items: tuple[Any, ...] = (key, pytnl.devices.Host)
        else:
            # DenseMatrix[float, Host] / DenseMatrix[float, Cuda]
            items = key

        value_type = items[0]
        if value_type is not float:
            raise TypeError(f"DenseMatrix supports only float value type, got {value_type.__name__}")
        return self._get_cpp_class(items)


class _DenseMatrixViewMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "DenseMatrixView"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(self, key: type[float], /) -> type[pytnl._matrices.DenseMatrixView_float]: ...

    @overload
    def __getitem__(
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_float]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixView_float]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[type[Any], type[Any]],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            # DenseMatrixView[float] → default device=Host
            items: tuple[Any, ...] = (key, pytnl.devices.Host)
        else:
            # DenseMatrixView[float, Host] / DenseMatrixView[float, Cuda]
            items = key

        value_type = items[0]
        if value_type is not float:
            raise TypeError(f"DenseMatrixView supports only float value type, got {value_type.__name__}")
        return self._get_cpp_class(items)


class DenseMatrix(metaclass=_DenseMatrixMeta):
    """
    Allows `DenseMatrix[value_type, device_type]` syntax to resolve to
    the appropriate C++ dense matrix class.

    Only `float` (C++ `double`) is supported as the value type. Other types
    like `int` or `complex` will raise `TypeError`.

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.

    Examples:
    - `DenseMatrix[float]` → DenseMatrix on Host
    - `DenseMatrix[float, devices.Cuda]` → DenseMatrix on Cuda
    """


class SparseMatrix(metaclass=_SparseMatrixMeta):
    """
    Allows `SparseMatrix[value_type, device_type, format]` syntax to resolve to
    the appropriate C++ sparse matrix class.

    Only `float` (C++ `double`) is supported as the value type. Other types
    like `int` or `complex` will raise `TypeError`.

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.
    The `format` argument is optional and defaults to `formats.CSR`. Supported
    formats: `formats.CSR`, `formats.Ellpack`, `formats.SlicedEllpack`.

    In the two-argument form, the second argument is always a device type.
    To select a non-default format, use the three-argument form.

    Examples:
    - `SparseMatrix[float]` → CSR matrix on Host
    - `SparseMatrix[float, devices.Cuda]` → CSR matrix on Cuda
    - `SparseMatrix[float, devices.Host, formats.Ellpack]` → Ellpack matrix on Host
    - `SparseMatrix[float, devices.Cuda, formats.SlicedEllpack]` → SlicedEllpack on Cuda
    """


class DenseMatrixView(metaclass=_DenseMatrixViewMeta):
    """
    Allows `DenseMatrixView[value_type, device_type]` syntax to resolve to
    the appropriate C++ dense matrix view class.

    Only `float` (C++ `double`) is supported as the value type. Other types
    like `int` or `complex` will raise `TypeError`.

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.

    Examples:
    - `DenseMatrixView[float]` → DenseMatrixView on Host
    - `DenseMatrixView[float, devices.Cuda]` → DenseMatrixView on Cuda
    """


class SparseMatrixView(metaclass=_SparseMatrixViewMeta):
    """
    Allows `SparseMatrixView[value_type, device_type, format]` syntax to resolve to
    the appropriate C++ sparse matrix view class.

    Only `float` (C++ `double`) is supported as the value type. Other types
    like `int` or `complex` will raise `TypeError`.

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.
    The `format` argument is optional and defaults to `formats.CSR`. Supported
    formats: `formats.CSR`, `formats.Ellpack`, `formats.SlicedEllpack`.

    In the two-argument form, the second argument is always a device type.
    To select a non-default format, use the three-argument form.

    Examples:
    - `SparseMatrixView[float]` → CSR view on Host
    - `SparseMatrixView[float, devices.Cuda]` → CSR view on Cuda
    - `SparseMatrixView[float, devices.Host, formats.Ellpack]` → Ellpack view on Host
    - `SparseMatrixView[float, devices.Cuda, formats.SlicedEllpack]` → SlicedEllpack view on Cuda
    """


# Re-export helper types from the C++ module
DenseMatrixRowView = pytnl._matrices.DenseMatrixRowView
DenseMatrixConstRowView = pytnl._matrices.DenseMatrixConstRowView
SparseMatrixRowView = pytnl._matrices.SparseMatrixRowView
SparseMatrixConstRowView = pytnl._matrices.SparseMatrixConstRowView

copySparseMatrix = pytnl._matrices.copySparseMatrix

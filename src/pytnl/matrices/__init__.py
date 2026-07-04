from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import pytnl._matrices
import pytnl._meta
import pytnl.devices
from pytnl._matrices import ElementsOrganization, formats

if TYPE_CHECKING:
    # This is an optional module - at runtime it is lazy-imported in
    # `CPPClassTemplate`, for type checking there must be the import statement.
    import pytnl._matrices_cuda as _matrices_cuda  # type: ignore[import-not-found, unused-ignore]

__all__ = [
    "DenseMatrix",
    "DenseMatrixConstRowView",
    "DenseMatrixRowView",
    "DenseMatrixView",
    "ElementsOrganization",
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
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_float_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
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
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.CSR]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_float_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.Ellpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_float_Ellpack]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
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

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_complex_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_complex_CSR]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_complex_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], type[formats.CSR]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_complex_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], type[formats.Ellpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_complex_Ellpack]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], type[formats.SlicedEllpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrix_complex_SlicedEllpack]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], type[formats.CSR]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_complex_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], type[formats.Ellpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_complex_Ellpack]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], type[formats.SlicedEllpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrix_complex_SlicedEllpack]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[type[Any], type[Any]] | tuple[type[Any], type[Any], type[Any]],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            items: tuple[Any, ...] = (key, pytnl.devices.Host, formats.CSR)
        elif len(key) == 2:
            items = (key[0], key[1], formats.CSR)
        else:
            # SparseMatrix[float, Host, formats.CSR]
            items = key

        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"SparseMatrix supports only float or complex value type, got {value_type.__name__}")
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
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_float_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
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
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.CSR]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_float_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], type[formats.Ellpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_float_Ellpack]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
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

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_complex_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_complex_CSR]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_complex_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], type[formats.CSR]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_complex_CSR]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], type[formats.Ellpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_complex_Ellpack]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], type[formats.SlicedEllpack]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixView_complex_SlicedEllpack]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], type[formats.CSR]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_complex_CSR]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], type[formats.Ellpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_complex_Ellpack]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], type[formats.SlicedEllpack]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixView_complex_SlicedEllpack]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[type[Any], type[Any]] | tuple[type[Any], type[Any], type[Any]],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            items: tuple[Any, ...] = (key, pytnl.devices.Host, formats.CSR)
        elif len(key) == 2:
            items = (key[0], key[1], formats.CSR)
        else:
            items = key

        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"SparseMatrixView supports only float or complex value type, got {value_type.__name__}")
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
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_float_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrix_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_float_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrix_float_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrix_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_complex_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrix_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrix_complex_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrix_complex_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrix_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[Any, ...],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            items = (key, pytnl.devices.Host)
            org = ElementsOrganization.RowMajorOrder
        elif len(key) == 2:
            items = key
            org = ElementsOrganization.RowMajorOrder if items[1] is pytnl.devices.Host else ElementsOrganization.ColumnMajorOrder
        elif len(key) == 3:
            items = (key[0], key[1])
            org = key[2]
        else:
            raise TypeError(f"DenseMatrix must be subscripted with 1, 2, or 3 arguments, got {len(key)}")

        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"DenseMatrix supports only float or complex value type, got {value_type.__name__}")

        if org not in (ElementsOrganization.RowMajorOrder, ElementsOrganization.ColumnMajorOrder):
            raise TypeError(f"Unsupported organization: {org}")

        module, base_name = self._validate_params(items)
        org_name = "RowMajor" if org is ElementsOrganization.RowMajorOrder else "ColumnMajor"
        class_name = f"{base_name}_{org_name}"

        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_name}' not found in module '{module.__name__}'.")
        return cast(type[Any], getattr(module, class_name))


class _DenseMatrixViewMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "DenseMatrixView"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_float_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixView_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_float_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixView_float_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixView_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixView_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixView_complex_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixView_complex_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixView_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[Any, ...],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            items = (key, pytnl.devices.Host)
            org = ElementsOrganization.RowMajorOrder
        elif len(key) == 2:
            items = key
            org = ElementsOrganization.RowMajorOrder if items[1] is pytnl.devices.Host else ElementsOrganization.ColumnMajorOrder
        elif len(key) == 3:
            items = (key[0], key[1])
            org = key[2]
        else:
            raise TypeError(f"DenseMatrixView must be subscripted with 1, 2, or 3 arguments, got {len(key)}")

        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"DenseMatrixView supports only float or complex value type, got {value_type.__name__}")

        if org not in (ElementsOrganization.RowMajorOrder, ElementsOrganization.ColumnMajorOrder):
            raise TypeError(f"Unsupported organization: {org}")

        module, base_name = self._validate_params(items)
        org_name = "RowMajor" if org is ElementsOrganization.RowMajorOrder else "ColumnMajor"
        class_name = f"{base_name}_{org_name}"

        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_name}' not found in module '{module.__name__}'.")
        return cast(type[Any], getattr(module, class_name))


class DenseMatrix(metaclass=_DenseMatrixMeta):
    """
    Allows `DenseMatrix[value_type, device_type, organization]` syntax to resolve to
    the appropriate C++ dense matrix class.

    Supported value types: `float` (C++ `double`) and `complex` (C++
    `std::complex<double>` on Host, `TNL::Arithmetics::Complex<double>` on Cuda).

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.
    The `organization` argument is optional and defaults to RowMajor on Host, ColumnMajor on Cuda.

    Examples:
    - `DenseMatrix[float]` → DenseMatrix on Host (RowMajor)
    - `DenseMatrix[float, devices.Cuda]` → DenseMatrix on Cuda (ColumnMajor)
    - `DenseMatrix[float, devices.Host, ElementsOrganization.ColumnMajorOrder]` → DenseMatrix on Host (ColumnMajor)
    - `DenseMatrix[complex]` → DenseMatrix on Host (RowMajor) with complex values
    - `DenseMatrix[complex, devices.Cuda]` → DenseMatrix on Cuda (ColumnMajor) with complex values
    """


class SparseMatrix(metaclass=_SparseMatrixMeta):
    """
    Allows `SparseMatrix[value_type, device_type, format]` syntax to resolve to
    the appropriate C++ sparse matrix class.

    Supported value types: `float` (C++ `double`) and `complex` (C++
    `std::complex<double>` on Host, `TNL::Arithmetics::Complex<double>` on Cuda).

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
    - `SparseMatrix[complex]` → CSR matrix on Host with complex values
    - `SparseMatrix[complex, devices.Cuda, formats.Ellpack]` → Ellpack on Cuda with complex values
    """


class DenseMatrixView(metaclass=_DenseMatrixViewMeta):
    """
    Allows `DenseMatrixView[value_type, device_type, organization]` syntax to resolve to
    the appropriate C++ dense matrix view class.

    Supported value types: `float` (C++ `double`) and `complex` (C++
    `std::complex<double>` on Host, `TNL::Arithmetics::Complex<double>` on Cuda).

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.
    The `organization` argument is optional and defaults to RowMajor on Host, ColumnMajor on Cuda.

    Examples:
    - `DenseMatrixView[float]` → DenseMatrixView on Host (RowMajor)
    - `DenseMatrixView[float, devices.Cuda]` → DenseMatrixView on Cuda (ColumnMajor)
    - `DenseMatrixView[float, devices.Host, ElementsOrganization.ColumnMajorOrder]` → DenseMatrixView on Host (ColumnMajor)
    - `DenseMatrixView[complex]` → DenseMatrixView on Host (RowMajor) with complex values
    - `DenseMatrixView[complex, devices.Cuda]` → DenseMatrixView on Cuda (ColumnMajor) with complex values
    """


class SparseMatrixView(metaclass=_SparseMatrixViewMeta):
    """
    Allows `SparseMatrixView[value_type, device_type, format]` syntax to resolve to
    the appropriate C++ sparse matrix view class.

    Supported value types: `float` (C++ `double`) and `complex` (C++
    `std::complex<double>` on Host, `TNL::Arithmetics::Complex<double>` on Cuda).

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
    - `SparseMatrixView[complex]` → CSR view on Host with complex values
    - `SparseMatrixView[complex, devices.Cuda, formats.Ellpack]` → Ellpack view on Cuda with complex values
    """


class _SparseMatrixRowViewMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "SparseMatrixRowView"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float] | tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixRowView_float]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixRowView_float]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex] | tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixRowView_complex]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixRowView_complex]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[bool | pytnl._meta.VT] | tuple[type[bool | pytnl._meta.VT], type[pytnl._meta.DT]],
        /,
    ) -> type[Any]:
        if isinstance(key, tuple):
            items = key
        else:
            items = (key, pytnl.devices.Host)
        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"SparseMatrixRowView supports only float or complex value type, got {value_type.__name__}")
        return self._get_cpp_class(items)


class _SparseMatrixConstRowViewMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "SparseMatrixConstRowView"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float] | tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixConstRowView_float]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixConstRowView_float]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex] | tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.SparseMatrixConstRowView_complex]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.SparseMatrixConstRowView_complex]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[bool | pytnl._meta.VT] | tuple[type[bool | pytnl._meta.VT], type[pytnl._meta.DT]],
        /,
    ) -> type[Any]:
        if isinstance(key, tuple):
            items = key
        else:
            items = (key, pytnl.devices.Host)
        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"SparseMatrixConstRowView supports only float or complex value type, got {value_type.__name__}")
        return self._get_cpp_class(items)


class _DenseMatrixRowViewMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "DenseMatrixRowView"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_float_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixRowView_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_float_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixRowView_float_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixRowView_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixRowView_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixRowView_complex_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixRowView_complex_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixRowView_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[Any, ...],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            items = (key, pytnl.devices.Host)
            org = ElementsOrganization.RowMajorOrder
        elif len(key) == 2:
            items = key
            org = ElementsOrganization.RowMajorOrder if items[1] is pytnl.devices.Host else ElementsOrganization.ColumnMajorOrder
        elif len(key) == 3:
            items = (key[0], key[1])
            org = key[2]
        else:
            raise TypeError(f"DenseMatrixRowView must be subscripted with 1, 2, or 3 arguments, got {len(key)}")

        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"DenseMatrixRowView supports only float or complex value type, got {value_type.__name__}")

        if org not in (ElementsOrganization.RowMajorOrder, ElementsOrganization.ColumnMajorOrder):
            raise TypeError(f"Unsupported organization: {org}")

        module, base_name = self._validate_params(items)
        org_name = "RowMajor" if org is ElementsOrganization.RowMajorOrder else "ColumnMajor"
        class_name = f"{base_name}_{org_name}"

        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_name}' not found in module '{module.__name__}'.")
        return cast(type[Any], getattr(module, class_name))


class _DenseMatrixConstRowViewMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._matrices
    _class_prefix = "DenseMatrixConstRowView"
    _template_parameters = (
        ("value_type", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[float],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_float_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixConstRowView_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_float_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[float], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_float_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixConstRowView_float_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[float], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixConstRowView_float_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: type[complex],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixConstRowView_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_complex_RowMajor]: ...

    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self,
        key: tuple[type[complex], type[pytnl.devices.Host], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[pytnl._matrices.DenseMatrixConstRowView_complex_ColumnMajor]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.RowMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixConstRowView_complex_RowMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[complex], type[pytnl.devices.Cuda], Literal[ElementsOrganization.ColumnMajorOrder]],
        /,
    ) -> type[_matrices_cuda.DenseMatrixConstRowView_complex_ColumnMajor]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: type[Any] | tuple[Any, ...],
        /,
    ) -> type[Any]:
        if not isinstance(key, tuple):
            items = (key, pytnl.devices.Host)
            org = ElementsOrganization.RowMajorOrder
        elif len(key) == 2:
            items = key
            org = ElementsOrganization.RowMajorOrder if items[1] is pytnl.devices.Host else ElementsOrganization.ColumnMajorOrder
        elif len(key) == 3:
            items = (key[0], key[1])
            org = key[2]
        else:
            raise TypeError(f"DenseMatrixConstRowView must be subscripted with 1, 2, or 3 arguments, got {len(key)}")

        value_type = items[0]
        if value_type not in (float, complex):
            raise TypeError(f"DenseMatrixConstRowView supports only float or complex value type, got {value_type.__name__}")

        if org not in (ElementsOrganization.RowMajorOrder, ElementsOrganization.ColumnMajorOrder):
            raise TypeError(f"Unsupported organization: {org}")

        module, base_name = self._validate_params(items)
        org_name = "RowMajor" if org is ElementsOrganization.RowMajorOrder else "ColumnMajor"
        class_name = f"{base_name}_{org_name}"

        if not hasattr(module, class_name):
            raise ValueError(f"Class '{class_name}' not found in module '{module.__name__}'.")
        return cast(type[Any], getattr(module, class_name))


class SparseMatrixRowView(metaclass=_SparseMatrixRowViewMeta):
    """
    Allows `SparseMatrixRowView[value_type, device_type]` syntax to resolve to
    the appropriate C++ sparse matrix row view class.

    All sparse formats (CSR, Ellpack, SlicedEllpack) share the same RowView type,
    so no format parameter is needed.

    Supported value types: `float` and `complex`.

    Examples:
    - `SparseMatrixRowView[float]` → Host row view for float matrices
    - `SparseMatrixRowView[complex, devices.Cuda]` → Cuda row view for complex matrices
    """


class SparseMatrixConstRowView(metaclass=_SparseMatrixConstRowViewMeta):
    """
    Allows `SparseMatrixConstRowView[value_type, device_type]` syntax to resolve to
    the appropriate C++ sparse matrix const row view class.

    Examples:
    - `SparseMatrixConstRowView[float]` → Host const row view for float matrices
    - `SparseMatrixConstRowView[complex, devices.Cuda]` → Cuda const row view for complex matrices
    """


class DenseMatrixRowView(metaclass=_DenseMatrixRowViewMeta):
    """
    Allows `DenseMatrixRowView[value_type, device_type, organization]` syntax to resolve to
    the appropriate C++ dense matrix row view class.

    Examples:
    - `DenseMatrixRowView[float]` → Host RowMajor row view
    - `DenseMatrixRowView[float, devices.Cuda]` → Cuda ColumnMajor row view
    - `DenseMatrixRowView[complex, devices.Host, ElementsOrganization.ColumnMajorOrder]` → Host ColumnMajor
    """


class DenseMatrixConstRowView(metaclass=_DenseMatrixConstRowViewMeta):
    """
    Allows `DenseMatrixConstRowView[value_type, device_type, organization]` syntax to resolve to
    the appropriate C++ dense matrix const row view class.

    Examples:
    - `DenseMatrixConstRowView[float]` → Host RowMajor const row view
    - `DenseMatrixConstRowView[complex, devices.Cuda]` → Cuda ColumnMajor const row view
    """


def copySparseMatrix(destination: Any, source: Any) -> None:  # noqa: ANN401
    """
    Copy a sparse matrix to another, possibly different, format or device.

    Dispatches to the Host or CUDA backend based on the source and
    destination matrix modules. Cross-device copies (Host <-> Cuda) are
    dispatched to the CUDA module, which has both Host and Cuda overloads.

    Both source and destination must have the same value type (float or
    complex). Cross-value-type copies are not supported.

    Cross-device copies are only available for float matrices. Complex
    matrices cannot be copied across devices because Host uses
    ``std::complex<double>`` and Cuda uses
    ``TNL::Arithmetics::Complex<double>``, which are distinct C++ types.
    """
    src_is_cuda = type(source).__module__.startswith("pytnl._matrices_cuda")
    dst_is_cuda = type(destination).__module__.startswith("pytnl._matrices_cuda")

    if src_is_cuda != dst_is_cuda and ("complex" in type(source).__name__ or "complex" in type(destination).__name__):
        raise NotImplementedError(
            "Cross-device copySparseMatrix is not available for complex matrices (Host uses std::complex, Cuda uses TNL::Arithmetics::Complex)."
        )

    if src_is_cuda or dst_is_cuda:
        import pytnl._matrices_cuda as _matrices_cuda  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415

        _matrices_cuda.copySparseMatrix(destination, source)  # pyright: ignore[reportUnknownMemberType]
    else:
        pytnl._matrices.copySparseMatrix(destination, source)

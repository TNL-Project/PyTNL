from __future__ import annotations  # noqa: I001

from typing import TYPE_CHECKING, Any, Literal, overload

import pytnl._meshes
import pytnl._meta
import pytnl.devices

# Import objects where Pythonizations are not needed
from pytnl._meshes import (
    XMLVTK,
    MeshReader,
    PVTUReader,
    VTIReader,
    VTKCellGhostTypes,
    VTKDataType,
    VTKEntityShape,
    VTKFileFormat,
    VTKPointGhostTypes,
    VTKReader,
    VTKTypesArrayType,
    VTUReader,
    distributeFaces,
    getMeshReader,
    resolveAndLoadMesh,
    resolveMeshType,
)

# Import mesh types
# TODO: make some Pythonization for this
from pytnl._meshes import (
    DistributedMeshOfEdges,
    DistributedMeshOfHexahedrons,
    DistributedMeshOfQuadrangles,
    DistributedMeshOfTetrahedrons,
    DistributedMeshOfTriangles,
    MeshOfEdges,
    MeshOfHexahedrons,
    MeshOfPolygons,
    MeshOfPolyhedrons,
    MeshOfQuadrangles,
    MeshOfTetrahedrons,
    MeshOfTriangles,
)

# Import type aliases/variables
from pytnl._meta import DIMS, DT

if TYPE_CHECKING:
    # This is an optional module - at runtime it is lazy-imported in
    # `CPPClassTemplate`, for type checking there must be the import statement.
    import pytnl._meshes_cuda as _meshes_cuda  # type: ignore[import-not-found, unused-ignore]

__all__ = [
    "XMLVTK",
    "DistributedMeshOfEdges",
    "DistributedMeshOfHexahedrons",
    "DistributedMeshOfQuadrangles",
    "DistributedMeshOfTetrahedrons",
    "DistributedMeshOfTriangles",
    "Grid",
    "MeshOfEdges",
    "MeshOfHexahedrons",
    "MeshOfPolygons",
    "MeshOfPolyhedrons",
    "MeshOfQuadrangles",
    "MeshOfTetrahedrons",
    "MeshOfTriangles",
    "MeshReader",
    "PVTUReader",
    "PVTUWriter",
    "VTIReader",
    "VTIWriter",
    "VTKCellGhostTypes",
    "VTKDataType",
    "VTKEntityShape",
    "VTKFileFormat",
    "VTKPointGhostTypes",
    "VTKReader",
    "VTKTypesArrayType",
    "VTKWriter",
    "VTUReader",
    "VTUWriter",
    "distributeFaces",
    "getMeshReader",
    "resolveAndLoadMesh",
    "resolveMeshType",
]


class _GridMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._meshes
    _class_prefix = "Grid"
    _template_parameters = (
        ("dimension", int),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(
        self,
        key: Literal[1] | tuple[Literal[1], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._meshes.Grid_1]: ...

    @overload
    def __getitem__(
        self,
        key: Literal[2] | tuple[Literal[2], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._meshes.Grid_2]: ...

    @overload
    def __getitem__(
        self,
        key: Literal[3] | tuple[Literal[3], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._meshes.Grid_3]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[Literal[1], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_meshes_cuda.Grid_1]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[Literal[2], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_meshes_cuda.Grid_2]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[Literal[3], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_meshes_cuda.Grid_3]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(
        self,
        key: DIMS | tuple[DIMS, type[DT]],
        /,
    ) -> type[Any]:
        if isinstance(key, tuple):
            items = key
        else:
            # make a tuple of arguments, use host as the default device
            items = (key, pytnl.devices.Host)
        return self._get_cpp_class(items)


class Grid(metaclass=_GridMeta):
    """
    Allows `Grid[dimension, device_type]` syntax to resolve to the appropriate
    C++ `Grid` class.

    This class provides a Python interface to C++ orthogonal grids.

    The `device_type` argument is optional and defaults to `pytnl.devices.host`.

    Examples:
    - `Grid[1]` → `_meshes.Grid_1`
    - `Grid[2, devices.cuda]` → `_meshes_cuda.Grid_2`
    - `Grid[3, devices.host]` → `_meshes.Grid_3`
    """


class _VTIWriterMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._meshes
    _class_prefix = "VTIWriter"
    _template_parameters = (("grid_type", type),)
    _dispatch_same_module_parameter = "grid_type"

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_1],
        /,
    ) -> type[pytnl._meshes.VTIWriter_Grid_1]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_2],
        /,
    ) -> type[pytnl._meshes.VTIWriter_Grid_2]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_3],
        /,
    ) -> type[pytnl._meshes.VTIWriter_Grid_3]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_1],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTIWriter_Grid_1]: ...  # pyright: ignore

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, overload-cannot-match, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_2],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTIWriter_Grid_2]: ...  # pyright: ignore

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, overload-cannot-match, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_3],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTIWriter_Grid_3]: ...  # pyright: ignore

    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: type[  # pyright: ignore
            pytnl._meshes.Grid_1
            | pytnl._meshes.Grid_2
            | pytnl._meshes.Grid_3
            | _meshes_cuda.Grid_1  # pyright: ignore
            | _meshes_cuda.Grid_2  # pyright: ignore
            | _meshes_cuda.Grid_3,  # pyright: ignore
        ],
        /,
    ) -> type[Any]:
        items = (key,)  # pyright: ignore
        return self._get_cpp_class(items)  # pyright: ignore


class VTIWriter(metaclass=_VTIWriterMeta):
    """
    Allows `VTIWriter[grid_type]` syntax to resolve to the appropriate C++ `VTIWriter` class.

    This class provides a Python interface to C++ writers for orthogonal grids.

    Example:
    - `VTIWriter[Grid[1]]` → `VTIWriter_Grid_1`
    - `VTIWriter[Grid[2]]` → `VTIWriter_Grid_2`
    - `VTIWriter[Grid[3]]` → `VTIWriter_Grid_3`
    """


class _VTUWriterMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._meshes
    _class_prefix = "VTUWriter"
    _template_parameters = (("mesh_type", type),)
    _dispatch_same_module_parameter = "mesh_type"

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_1],
        /,
    ) -> type[pytnl._meshes.VTUWriter_Grid_1]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_2],
        /,
    ) -> type[pytnl._meshes.VTUWriter_Grid_2]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_3],
        /,
    ) -> type[pytnl._meshes.VTUWriter_Grid_3]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfEdges],
        /,
    ) -> type[pytnl._meshes.VTUWriter_MeshOfEdges]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfHexahedrons],
        /,
    ) -> type[pytnl._meshes.VTUWriter_MeshOfHexahedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfPolygons],
        /,
    ) -> type[pytnl._meshes.VTUWriter_MeshOfPolygons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfPolyhedrons],
        /,
    ) -> type[pytnl._meshes.VTUWriter_MeshOfPolyhedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfQuadrangles],
        /,
    ) -> type[pytnl._meshes.VTUWriter_MeshOfQuadrangles]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfTetrahedrons],
        /,
    ) -> type[pytnl._meshes.VTUWriter_MeshOfTetrahedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfTriangles],
        /,
    ) -> type[pytnl._meshes.VTUWriter_MeshOfTriangles]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_1],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTUWriter_Grid_1]: ...  # pyright: ignore

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, overload-cannot-match, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_2],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTUWriter_Grid_2]: ...  # pyright: ignore

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, overload-cannot-match, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_3],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTUWriter_Grid_3]: ...  # pyright: ignore

    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: type[  # pyright: ignore
            pytnl._meshes.Grid_1
            | pytnl._meshes.Grid_2
            | pytnl._meshes.Grid_3
            | pytnl._meshes.MeshOfEdges
            | pytnl._meshes.MeshOfHexahedrons
            | pytnl._meshes.MeshOfPolygons
            | pytnl._meshes.MeshOfPolyhedrons
            | pytnl._meshes.MeshOfQuadrangles
            | pytnl._meshes.MeshOfTetrahedrons
            | pytnl._meshes.MeshOfTriangles
            | _meshes_cuda.Grid_1  # pyright: ignore
            | _meshes_cuda.Grid_2  # pyright: ignore
            | _meshes_cuda.Grid_3  # pyright: ignore
        ],
        /,
    ) -> type[Any]:
        items = (key,)  # pyright: ignore
        return self._get_cpp_class(items)  # pyright: ignore


class VTUWriter(metaclass=_VTUWriterMeta):
    """
    Allows `VTUWriter[mesh_type]` syntax to resolve to the appropriate C++ `VTUWriter` class.

    This class provides a Python interface to C++ writers for orthogonal grids.

    Examples:
    - `VTUWriter[MeshOfEdges]` → `VTUWriter_MeshOfEdges`
    - `VTUWriter[MeshOfPolygons]` → `VTUWriter_MeshOfPolygons`
    """


class _VTKWriterMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._meshes
    _class_prefix = "VTKWriter"
    _template_parameters = (("mesh_type", type),)
    _dispatch_same_module_parameter = "mesh_type"

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_1],
        /,
    ) -> type[pytnl._meshes.VTKWriter_Grid_1]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_2],
        /,
    ) -> type[pytnl._meshes.VTKWriter_Grid_2]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.Grid_3],
        /,
    ) -> type[pytnl._meshes.VTKWriter_Grid_3]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfEdges],
        /,
    ) -> type[pytnl._meshes.VTKWriter_MeshOfEdges]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfHexahedrons],
        /,
    ) -> type[pytnl._meshes.VTKWriter_MeshOfHexahedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfPolygons],
        /,
    ) -> type[pytnl._meshes.VTKWriter_MeshOfPolygons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfPolyhedrons],
        /,
    ) -> type[pytnl._meshes.VTKWriter_MeshOfPolyhedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfQuadrangles],
        /,
    ) -> type[pytnl._meshes.VTKWriter_MeshOfQuadrangles]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfTetrahedrons],
        /,
    ) -> type[pytnl._meshes.VTKWriter_MeshOfTetrahedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfTriangles],
        /,
    ) -> type[pytnl._meshes.VTKWriter_MeshOfTriangles]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_1],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTKWriter_Grid_1]: ...  # pyright: ignore

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, overload-cannot-match, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_2],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTKWriter_Grid_2]: ...  # pyright: ignore

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, overload-cannot-match, unused-ignore]
        self,
        key: type[_meshes_cuda.Grid_3],  # pyright: ignore
        /,
    ) -> type[_meshes_cuda.VTKWriter_Grid_3]: ...  # pyright: ignore

    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: type[  # pyright: ignore
            pytnl._meshes.Grid_1
            | pytnl._meshes.Grid_2
            | pytnl._meshes.Grid_3
            | pytnl._meshes.MeshOfEdges
            | pytnl._meshes.MeshOfHexahedrons
            | pytnl._meshes.MeshOfPolygons
            | pytnl._meshes.MeshOfPolyhedrons
            | pytnl._meshes.MeshOfQuadrangles
            | pytnl._meshes.MeshOfTetrahedrons
            | pytnl._meshes.MeshOfTriangles
            | _meshes_cuda.Grid_1  # pyright: ignore
            | _meshes_cuda.Grid_2  # pyright: ignore
            | _meshes_cuda.Grid_3  # pyright: ignore
        ],
        /,
    ) -> type[Any]:
        items = (key,)  # pyright: ignore
        return self._get_cpp_class(items)  # pyright: ignore


class VTKWriter(metaclass=_VTKWriterMeta):
    """
    Allows `VTKWriter[mesh_type]` syntax to resolve to the appropriate C++ `VTKWriter` class.

    This class provides a Python interface to C++ writers for orthogonal grids.

    Examples:
    - `VTKWriter[MeshOfEdges]` → `VTKWriter_MeshOfEdges`
    - `VTKWriter[MeshOfPolygons]` → `VTKWriter_MeshOfPolygons`
    """


class _PVTUWriterMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._meshes
    _class_prefix = "PVTUWriter"
    _template_parameters = (("mesh_type", type),)
    _dispatch_same_module_parameter = "mesh_type"

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfEdges],
        /,
    ) -> type[pytnl._meshes.PVTUWriter_MeshOfEdges]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfHexahedrons],
        /,
    ) -> type[pytnl._meshes.PVTUWriter_MeshOfHexahedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfQuadrangles],
        /,
    ) -> type[pytnl._meshes.PVTUWriter_MeshOfQuadrangles]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfTetrahedrons],
        /,
    ) -> type[pytnl._meshes.PVTUWriter_MeshOfTetrahedrons]: ...

    @overload
    def __getitem__(
        self,
        key: type[pytnl._meshes.MeshOfTriangles],
        /,
    ) -> type[pytnl._meshes.PVTUWriter_MeshOfTriangles]: ...

    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: type[  # pyright: ignore
            pytnl._meshes.MeshOfEdges
            | pytnl._meshes.MeshOfHexahedrons
            | pytnl._meshes.MeshOfQuadrangles
            | pytnl._meshes.MeshOfTetrahedrons
            | pytnl._meshes.MeshOfTriangles
        ],
        /,
    ) -> type[Any]:
        items = (key,)  # pyright: ignore
        return self._get_cpp_class(items)  # pyright: ignore


class PVTUWriter(metaclass=_PVTUWriterMeta):
    """
    Allows `PVTUWriter[mesh_type]` syntax to resolve to the appropriate C++ `PVTUWriter` class.

    This class provides a Python interface to C++ writers for orthogonal grids.

    Example:
    - `PVTUWriter[MeshOfEdges]` → `PVTUWriter_MeshOfEdges`
    - `PVTUWriter[MeshOfPolygons]` → `PVTUWriter_MeshOfPolygons`
    """

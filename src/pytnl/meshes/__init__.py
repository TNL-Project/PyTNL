from typing import Any, Literal, overload, override  # noqa: I001

import pytnl._meshes
import pytnl._meta

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
from pytnl._meta import DIMS

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
    _template_parameters = (("dimension", int),)

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload  # type: ignore[override]
    def __getitem__(self, items: Literal[1]) -> type[pytnl._meshes.Grid_1]: ...

    @overload
    def __getitem__(self, items: Literal[2]) -> type[pytnl._meshes.Grid_2]: ...

    @overload
    def __getitem__(self, items: Literal[3]) -> type[pytnl._meshes.Grid_3]: ...

    @override
    def __getitem__(self, items: DIMS) -> type[Any]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return super().__getitem__(items)


class Grid(metaclass=_GridMeta):
    """
    Allows `Grid[dimension]` syntax to resolve to the appropriate C++ `Grid` class.

    This class provides a Python interface to C++ orthogonal grids.

    Examples:
    - `Grid[1]` → `Grid_1`
    - `Grid[2]` → `Grid_2`
    - `Grid[3]` → `Grid_3`
    """


class _VTIWriterMeta(pytnl._meta.CPPClassTemplate):
    _cpp_module = pytnl._meshes
    _class_prefix = "VTIWriter"
    _template_parameters = (("grid_type", type),)

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload  # type: ignore[override]
    def __getitem__(self, items: type[pytnl._meshes.Grid_1]) -> type[pytnl._meshes.VTIWriter_Grid_1]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.Grid_2]) -> type[pytnl._meshes.VTIWriter_Grid_2]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.Grid_3]) -> type[pytnl._meshes.VTIWriter_Grid_3]: ...

    @override
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, items: type[pytnl._meshes.Grid_1 | pytnl._meshes.Grid_2 | pytnl._meshes.Grid_3]
    ) -> type[Any]:
        return super().__getitem__(items)


class VTIWriter(metaclass=_VTIWriterMeta):
    """
    Allows `VTIWriter[Grid[dimension]]` syntax to resolve to the appropriate C++ `VTIWriter` class.

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

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload  # type: ignore[override]
    def __getitem__(self, items: type[pytnl._meshes.Grid_1]) -> type[pytnl._meshes.VTUWriter_Grid_1]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.Grid_2]) -> type[pytnl._meshes.VTUWriter_Grid_2]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.Grid_3]) -> type[pytnl._meshes.VTUWriter_Grid_3]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfEdges]) -> type[pytnl._meshes.VTUWriter_MeshOfEdges]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfHexahedrons]) -> type[pytnl._meshes.VTUWriter_MeshOfHexahedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfPolygons]) -> type[pytnl._meshes.VTUWriter_MeshOfPolygons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfPolyhedrons]) -> type[pytnl._meshes.VTUWriter_MeshOfPolyhedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfQuadrangles]) -> type[pytnl._meshes.VTUWriter_MeshOfQuadrangles]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfTetrahedrons]) -> type[pytnl._meshes.VTUWriter_MeshOfTetrahedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfTriangles]) -> type[pytnl._meshes.VTUWriter_MeshOfTriangles]: ...

    @override
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        items: type[
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
        ],
    ) -> type[Any]:
        return super().__getitem__(items)


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

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload  # type: ignore[override]
    def __getitem__(self, items: type[pytnl._meshes.Grid_1]) -> type[pytnl._meshes.VTKWriter_Grid_1]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.Grid_2]) -> type[pytnl._meshes.VTKWriter_Grid_2]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.Grid_3]) -> type[pytnl._meshes.VTKWriter_Grid_3]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfEdges]) -> type[pytnl._meshes.VTKWriter_MeshOfEdges]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfHexahedrons]) -> type[pytnl._meshes.VTKWriter_MeshOfHexahedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfPolygons]) -> type[pytnl._meshes.VTKWriter_MeshOfPolygons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfPolyhedrons]) -> type[pytnl._meshes.VTKWriter_MeshOfPolyhedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfQuadrangles]) -> type[pytnl._meshes.VTKWriter_MeshOfQuadrangles]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfTetrahedrons]) -> type[pytnl._meshes.VTKWriter_MeshOfTetrahedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfTriangles]) -> type[pytnl._meshes.VTKWriter_MeshOfTriangles]: ...

    @override
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        items: type[
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
        ],
    ) -> type[Any]:
        return super().__getitem__(items)


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

    # NOTE: Python's typing `float` type accepts even `int` so the overloads
    # "overlap" and `float` must be carefully ordered last so that pyright
    # selects the first overload in a tie.
    # https://stackoverflow.com/a/62734976

    @overload  # type: ignore[override]
    def __getitem__(self, items: type[pytnl._meshes.MeshOfEdges]) -> type[pytnl._meshes.PVTUWriter_MeshOfEdges]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfHexahedrons]) -> type[pytnl._meshes.PVTUWriter_MeshOfHexahedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfQuadrangles]) -> type[pytnl._meshes.PVTUWriter_MeshOfQuadrangles]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfTetrahedrons]) -> type[pytnl._meshes.PVTUWriter_MeshOfTetrahedrons]: ...

    @overload
    def __getitem__(self, items: type[pytnl._meshes.MeshOfTriangles]) -> type[pytnl._meshes.PVTUWriter_MeshOfTriangles]: ...

    @override
    def __getitem__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        items: type[
            pytnl._meshes.MeshOfEdges
            | pytnl._meshes.MeshOfHexahedrons
            | pytnl._meshes.MeshOfQuadrangles
            | pytnl._meshes.MeshOfTetrahedrons
            | pytnl._meshes.MeshOfTriangles
        ],
    ) -> type[Any]:
        return super().__getitem__(items)


class PVTUWriter(metaclass=_PVTUWriterMeta):
    """
    Allows `PVTUWriter[mesh_type]` syntax to resolve to the appropriate C++ `PVTUWriter` class.

    This class provides a Python interface to C++ writers for orthogonal grids.

    Example:
    - `PVTUWriter[MeshOfEdges]` → `PVTUWriter_MeshOfEdges`
    - `PVTUWriter[MeshOfPolygons]` → `PVTUWriter_MeshOfPolygons`
    """

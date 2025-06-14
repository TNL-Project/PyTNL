from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

import pytnl.meshes
import pytnl.mpi

mpi4py: ModuleType | None
try:
    import mpi4py
    import mpi4py.MPI
except ImportError:
    mpi4py = None

# Global flags
TNL_DECOMPOSE_CMD = "tnl-decompose-mesh"
TNL_DECOMPOSE_FLAGS = "--ghost-levels 1"


# Mapping from file suffix to reader class
suffix_to_reader = {
    ".vtk": pytnl.meshes.VTKReader,
    ".vtu": pytnl.meshes.VTUReader,
}

# Mapping from topology directory to mesh and writer classes
mesh_writer_map = {
    "triangles": (pytnl.meshes.MeshOfTriangles, pytnl.meshes.VTKWriter_MeshOfTriangles, pytnl.meshes.VTUWriter_MeshOfTriangles),
    "triangles_2x2x2": (pytnl.meshes.MeshOfTriangles, pytnl.meshes.VTKWriter_MeshOfTriangles, pytnl.meshes.VTUWriter_MeshOfTriangles),
    "tetrahedrons": (pytnl.meshes.MeshOfTetrahedrons, pytnl.meshes.VTKWriter_MeshOfTetrahedrons, pytnl.meshes.VTUWriter_MeshOfTetrahedrons),
    "quadrangles": (pytnl.meshes.MeshOfQuadrangles, pytnl.meshes.VTKWriter_MeshOfQuadrangles, pytnl.meshes.VTUWriter_MeshOfQuadrangles),
    "hexahedrons": (pytnl.meshes.MeshOfHexahedrons, pytnl.meshes.VTKWriter_MeshOfHexahedrons, pytnl.meshes.VTUWriter_MeshOfHexahedrons),
    "polygons": (pytnl.meshes.MeshOfPolygons, pytnl.meshes.VTKWriter_MeshOfPolygons, pytnl.meshes.VTUWriter_MeshOfPolygons),
    "polyhedrons": (pytnl.meshes.MeshOfPolyhedrons, pytnl.meshes.VTKWriter_MeshOfPolyhedrons, pytnl.meshes.VTUWriter_MeshOfPolyhedrons),
}

# Define test cases with expected vertex and cell counts
test_cases = [
    # triangles
    ("triangles/mrizka_1.vtk", 142, 242),
    ("triangles/mrizka_1.vtu", 142, 242),
    # tetrahedrons
    ("tetrahedrons/cube1m_1.vtk", 395, 1312),
    ("tetrahedrons/cube1m_1.vtu", 395, 1312),
    # triangles_2x2x2
    ("triangles_2x2x2/original_with_metadata_and_cell_data.vtk", 9, 8),
    ("triangles_2x2x2/minimized_ascii.vtk", 9, 8),
    ("triangles_2x2x2/minimized_binary.vtk", 9, 8),
    ("triangles_2x2x2/version_5.1_ascii.vtk", 9, 8),
    ("triangles_2x2x2/version_5.1_binary.vtk", 9, 8),
    # quadrangles
    ("quadrangles/grid_2x3.vtk", 12, 6),
    ("quadrangles/grid_2x3.vtu", 12, 6),
    # hexahedrons
    ("hexahedrons/grid_2x3x4.vtk", 60, 24),
    ("hexahedrons/grid_2x3x4.vtu", 60, 24),
    # polygons
    ("polygons/unicorn.vtk", 193, 90),
    ("polygons/unicorn.vtu", 193, 90),
    # polyhedrons
    ("polyhedrons/two_polyhedra.vtk", 22, 2),
    ("polyhedrons/two_polyhedra.vtu", 22, 2),
    ("polyhedrons/cube1m_1.vtk", 2018, 395),
    ("polyhedrons/cube1m_1.vtu", 2018, 395),
]


# Generic function for testing mesh readers and writers
def _test_reader_writer(reader_class, writer_class, mesh, tmp_path):
    """
    Test that writing a mesh to a file with the given writer and reading it back
    with the given reader preserves the mesh structure.

    Parameters:
    - reader_class: The reader class (e.g., tnl.VTKReader)
    - writer_class: The writer class (e.g., tnl.VTKWriter_MeshOfTriangles)
    - mesh: A populated mesh instance to test
    - tmp_path: Pytest fixture for temporary directory
    """
    output_file = tmp_path / "test_mesh"

    # Write the mesh to a temporary file
    with open(output_file, "wb") as file:
        writer = writer_class(file)
        writer.writeMetadata(cycle=0, time=1.0)  # Write metadata header
        writer.writeCells(mesh)  # Write the mesh structure
        del writer  # Force flush

    # Check that the writer produced output
    output = output_file.read_bytes()
    if "VTKWriter" in writer_class.__name__:
        assert output.startswith(b"# vtk DataFile Version 5.1\n")
        assert output.count(b"# vtk DataFile Version 5.1\n") == 1
    else:
        assert output.startswith(b'<?xml version="1.0"?>\n<VTKFile type="')
        assert output.rstrip().endswith(b"</VTKFile>")

    # Read the mesh back from the file
    mesh_out = type(mesh)()
    reader = reader_class(str(output_file))
    reader.loadMesh(mesh_out)

    assert mesh_out == mesh


# Generic function for testing mesh readers and writers with data arrays
def _test_meshfunction(reader_class, writer_class, mesh, tmp_path, data_type="PointData"):
    """
    Tests writing and reading point/cell data arrays with the mesh.
    """
    if data_type == "PointData":
        n_points = mesh.getEntitiesCount(mesh.Vertex)
        scalar_data = list(range(n_points))
        vector_data = list(range(3 * n_points))
    else:
        n_cells = mesh.getEntitiesCount(mesh.Cell)
        scalar_data = list(range(n_cells))
        vector_data = list(range(3 * n_cells))

    output_file = tmp_path / "test_mesh"

    # Write mesh and data arrays
    with open(output_file, "wb") as file:
        writer = writer_class(file)
        writer.writeMetadata(cycle=42, time=3.14)
        writer.writeCells(mesh)
        if data_type == "PointData":
            writer.writePointData(scalar_data, "foo", 1)
            writer.writePointData(vector_data, "bar", 3)
        else:
            writer.writeCellData(scalar_data, "foo", 1)
            writer.writeCellData(vector_data, "bar", 3)
        del writer  # Force flush

    # Read mesh and data arrays
    mesh_out = type(mesh)()
    reader = reader_class(str(output_file))
    reader.loadMesh(mesh_out)

    if data_type == "PointData":
        scalar_data_in = reader.readPointData("foo")
        vector_data_in = reader.readPointData("bar")
    else:
        scalar_data_in = reader.readCellData("foo")
        vector_data_in = reader.readCellData("bar")

    assert scalar_data_in == scalar_data, f"{data_type} scalar data mismatch"
    assert vector_data_in == vector_data, f"{data_type} vector data mismatch"


# Parametrize test cases with file path, expected vertices, expected cells
@pytest.mark.parametrize("file_path, expected_vertices, expected_cells", test_cases)
def test_mesh_file(file_path, expected_vertices, expected_cells, tmp_path):
    data_dir = Path(__file__).parent / "data"
    full_path = (data_dir / file_path).resolve()
    suffix = full_path.suffix
    directory = full_path.parent.name

    # Get mesh class and writer class based on directory
    try:
        mesh_class, vtk_writer, vtu_writer = mesh_writer_map[directory]
    except KeyError:
        pytest.fail(f"Unsupported directory: {directory}")

    # Get reader class based on suffix
    try:
        reader_class = suffix_to_reader[suffix]
    except KeyError:
        pytest.fail(f"Unsupported file suffix: {suffix}")

    # Choose writer based on reader
    writer_class = vtk_writer if reader_class == pytnl.meshes.VTKReader else vtu_writer

    # Load mesh
    mesh = mesh_class()
    reader = reader_class(str(full_path))
    reader.loadMesh(mesh)

    # Check mesh entities
    assert mesh.getEntitiesCount(mesh.Vertex) == expected_vertices, f"Expected {expected_vertices} points in {file_path}"
    assert mesh.getEntitiesCount(mesh.Cell) == expected_cells, f"Expected {expected_cells} cells in {file_path}"

    # Round-trip tests
    _test_reader_writer(reader_class, writer_class, mesh, tmp_path)
    _test_meshfunction(reader_class, writer_class, mesh, tmp_path, "PointData")
    _test_meshfunction(reader_class, writer_class, mesh, tmp_path, "CellData")


# This test actually tests three functions:
# 1. getMeshReader - returns the MeshReader instance based on file extension
#       (does not call `reader.detectMesh` so it succeeds even for invalid file)
# 2. resolveMeshType - returns a `(reader, mesh)` pair where `reader` is initialized
#       with the given file name (using `getMeshReader`) and `mesh` is empty
# 3. resolveAndLoadMesh - same plus loads the mesh using `reader.loadMesh`
@pytest.mark.parametrize("file_path, expected_vertices, expected_cells", test_cases)
def test_resolveMeshType(file_path, expected_vertices, expected_cells, tmp_path):
    data_dir = Path(__file__).parent / "data"
    full_path = (data_dir / file_path).resolve()
    suffix = full_path.suffix
    directory = full_path.parent.name

    # Get mesh class and writer class based on directory
    try:
        mesh_class, vtk_writer, vtu_writer = mesh_writer_map[directory]
    except KeyError:
        pytest.fail(f"Unsupported directory: {directory}")

    # Get reader class based on suffix
    try:
        reader_class = suffix_to_reader[suffix]
    except KeyError:
        pytest.fail(f"Unsupported file suffix: {suffix}")

    # Test getMeshReader
    reader = pytnl.meshes.getMeshReader(f"invalid{suffix}")
    assert isinstance(reader, reader_class), reader
    reader = pytnl.meshes.getMeshReader(str(full_path))
    assert isinstance(reader, reader_class), reader

    # Test resolveMeshType
    with pytest.raises(RuntimeError):
        pytnl.meshes.resolveMeshType(f"invalid{suffix}")
    reader, mesh = pytnl.meshes.resolveMeshType(str(full_path))
    assert isinstance(reader, reader_class), reader
    assert isinstance(mesh, mesh_class), mesh
    # Check mesh entities
    assert mesh.getEntitiesCount(mesh.Vertex) == 0
    assert mesh.getEntitiesCount(mesh.Cell) == 0

    # Test resolveAndLoadMesh
    with pytest.raises(RuntimeError):
        pytnl.meshes.resolveAndLoadMesh(f"invalid{suffix}")
    reader, mesh = pytnl.meshes.resolveAndLoadMesh(str(full_path))
    assert isinstance(reader, reader_class), reader
    assert isinstance(mesh, mesh_class), mesh
    # Check mesh entities
    assert mesh.getEntitiesCount(mesh.Vertex) == expected_vertices, f"Expected {expected_vertices} points in {file_path}"
    assert mesh.getEntitiesCount(mesh.Cell) == expected_cells, f"Expected {expected_cells} cells in {file_path}"


# Test for PVTUReader and PVTUWriter (requires MPI)
@pytest.mark.mpi
@pytest.mark.skipif(not shutil.which("tnl-decompose-mesh"), reason="tnl-decompose-mesh is not available")
@pytest.mark.skipif(mpi4py is None or mpi4py.MPI.COMM_WORLD.Get_size() < 2, reason="Needs at least 2 MPI processes")
@pytest.mark.parametrize("file_path, expected_vertices, expected_cells", test_cases)
def test_pvtu_reader_writer(file_path: str, expected_vertices: int, expected_cells: int, tmp_path: Path):
    data_dir = Path(__file__).parent / "data"
    full_path = (data_dir / file_path).resolve()

    # Skip small meshes
    if expected_cells < 20:
        pytest.skip("not enough cells to decompose")

    assert mpi4py is not None
    comm = mpi4py.MPI.COMM_WORLD
    nproc = comm.Get_size()
    # rank = comm.Get_rank()

    # Decompose mesh first
    output_pvtu = tmp_path / "test.pvtu"
    cmd = f"{TNL_DECOMPOSE_CMD} --input-file {full_path} --output-file {output_pvtu} --subdomains {nproc} {TNL_DECOMPOSE_FLAGS}"
    subprocess.run(cmd, shell=True, check=True)

    # Load mesh and read data
    mesh = pytnl.mpi.DistributedMeshOfTriangles()
    local_mesh = mesh.getLocalMesh()
    reader = pytnl.mpi.PVTUReader(str(output_pvtu))
    reader.loadMesh(mesh)
    pytnl.mpi.distributeFaces(mesh)
    indices = reader.readCellData("GlobalIndex")

    assert len(indices) > 0
    assert min(indices) >= 0
    assert max(indices) > 0

    # Write test
    f = io.BytesIO()
    writer = pytnl.mpi.PVTUWriter_MeshOfTriangles(f)
    writer.writeCells(mesh)
    writer.writeMetadata(cycle=0, time=1.0)
    array = [42] * local_mesh.getEntitiesCount(mesh.Cell)
    writer.writePDataArray(array, "testArray", 1)
    for i in range(comm.Get_size()):
        path = writer.addPiece("pytnl_test.pvtu", i)
        assert path.endswith(f"/subdomain.{i}.vtu")
    del writer  # Force flush
    output = f.getvalue()

    assert output.startswith(b'<?xml version="1.0"?>\n<VTKFile type="PUnstructuredGrid"')
    assert output.count(b"<Piece") == comm.Get_size()

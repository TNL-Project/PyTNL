from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path
from types import ModuleType

import pytest

import tnl

mpi4py: ModuleType | None
try:
    import mpi4py
    import mpi4py.MPI
except ImportError:
    mpi4py = None

# Global flags
TNL_DECOMPOSE_CMD = "tnl-decompose-mesh"
TNL_DECOMPOSE_FLAGS = "--ghost-levels 1"


# Fixtures
@pytest.fixture
def input_vtk():
    return Path(__file__).parent / "data" / "triangles" / "mrizka_1.vtk"


@pytest.fixture
def input_vtu():
    return Path(__file__).parent / "data" / "triangles" / "mrizka_1.vtu"


@pytest.fixture
def test_vtk_file(tmp_path):
    return Path(tmp_path) / "test.vtk"


@pytest.fixture
def test_vtu_file(tmp_path):
    return Path(tmp_path) / "test.vtu"


# Tests for VTKReader and VTKWriter
def test_vtk_reader_writer(input_vtk: Path):
    mesh = tnl.MeshOfTriangles()
    reader = tnl.VTKReader(str(input_vtk))
    reader.loadMesh(mesh)
    assert mesh.getEntitiesCount(mesh.Cell) > 0

    f = io.BytesIO()
    writer = tnl.VTKWriter_MeshOfTriangles(f)
    writer.writeMetadata(cycle=0, time=1.0)
    writer.writeCells(mesh)
    array = [42] * mesh.getEntitiesCount(mesh.Cell)
    writer.writeDataArray(array, "testArray", 1)
    del writer  # Force flush

    output = f.getvalue()
    assert output.startswith(b"# vtk DataFile Version 5.1\n")
    assert output.count(b"# vtk DataFile Version 5.1\n") == 1


# Tests for VTUReader and VTUWriter
def test_vtu_reader_writer(input_vtu: Path):
    mesh = tnl.MeshOfTriangles()
    reader = tnl.VTUReader(str(input_vtu))
    reader.loadMesh(mesh)
    assert mesh.getEntitiesCount(mesh.Cell) > 0

    f = io.BytesIO()
    writer = tnl.VTUWriter_MeshOfTriangles(f)
    writer.writeMetadata(cycle=0, time=1.0)
    writer.writeCells(mesh)
    array = [42] * mesh.getEntitiesCount(mesh.Cell)
    writer.writeDataArray(array, "testArray", 1)
    del writer  # Force flush
    output = f.getvalue()

    assert output.startswith(b'<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid"')
    assert output.rstrip().endswith(b"</VTKFile>")
    assert b'<DataArray type="Int64" Name="testArray" NumberOfComponents="1"' in output


# Test for PVTUReader and PVTUWriter (requires MPI)
@pytest.mark.mpi
@pytest.mark.skipif(not shutil.which("tnl-decompose-mesh"), reason="tnl-decompose-mesh is not available")
@pytest.mark.skipif(mpi4py is None or mpi4py.MPI.COMM_WORLD.Get_size() < 2, reason="Needs at least 2 MPI processes")
def test_pvtu_reader_writer(input_vtu: Path, tmp_path: Path):
    output_pvtu = tmp_path / "test.pvtu"
    cmd = f"{TNL_DECOMPOSE_CMD} --input-file {input_vtu} --output-file {output_pvtu} --subdomains 4 {TNL_DECOMPOSE_FLAGS}"
    subprocess.run(cmd, shell=True, check=True)

    assert mpi4py is not None
    comm = mpi4py.MPI.COMM_WORLD
    # rank = comm.Get_rank()

    import tnl_mpi

    # Load mesh and read data
    mesh = tnl_mpi.DistributedMeshOfTriangles()
    local_mesh = mesh.getLocalMesh()
    reader = tnl_mpi.PVTUReader(str(output_pvtu))
    reader.loadMesh(mesh)
    tnl_mpi.distributeFaces(mesh)
    indices = reader.readCellData("GlobalIndex")

    assert len(indices) > 0
    assert min(indices) >= 0
    assert max(indices) >= 0

    # Write test
    f = io.BytesIO()
    writer = tnl_mpi.PVTUWriter_MeshOfTriangles(f)
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

import pytnl.containers
import pytnl.meshes


def test_typedefs():
    assert pytnl.meshes.Grid1D.CoordinatesType is pytnl.containers.StaticVector_1_int
    assert pytnl.meshes.Grid2D.CoordinatesType is pytnl.containers.StaticVector_2_int
    assert pytnl.meshes.Grid3D.CoordinatesType is pytnl.containers.StaticVector_3_int

    assert pytnl.meshes.Grid1D.PointType is pytnl.containers.StaticVector_1_float
    assert pytnl.meshes.Grid2D.PointType is pytnl.containers.StaticVector_2_float
    assert pytnl.meshes.Grid3D.PointType is pytnl.containers.StaticVector_3_float

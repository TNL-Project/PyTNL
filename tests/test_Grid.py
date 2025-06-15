import pytnl.containers
import pytnl.meshes


def test_typedefs():
    assert pytnl.meshes.Grid1D.CoordinatesType is pytnl.containers.StaticVector[1, int]
    assert pytnl.meshes.Grid2D.CoordinatesType is pytnl.containers.StaticVector[2, int]
    assert pytnl.meshes.Grid3D.CoordinatesType is pytnl.containers.StaticVector[3, int]

    assert pytnl.meshes.Grid1D.PointType is pytnl.containers.StaticVector[1, float]
    assert pytnl.meshes.Grid2D.PointType is pytnl.containers.StaticVector[2, float]
    assert pytnl.meshes.Grid3D.PointType is pytnl.containers.StaticVector[3, float]

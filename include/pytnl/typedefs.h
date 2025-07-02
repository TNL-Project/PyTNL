#pragma once

#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/DistributedMeshes/DistributedMesh.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Hexahedron.h>
#include <TNL/Meshes/Topologies/Polygon.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>
#include <TNL/Meshes/Topologies/Quadrangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/TypeResolver/BuildConfigTags.h>

using RealType = double;
using DeviceType = TNL::Devices::Host;
using IndexType = std::int64_t;
using ComplexType = std::complex< RealType >;

using Grid1D = TNL::Meshes::Grid< 1, RealType, DeviceType, IndexType >;
using Grid2D = TNL::Meshes::Grid< 2, RealType, DeviceType, IndexType >;
using Grid3D = TNL::Meshes::Grid< 3, RealType, DeviceType, IndexType >;

using LocalIndexType = short int;
template< typename Topology >
using DefaultMeshTemplate =
   TNL::Meshes::Mesh< TNL::Meshes::DefaultConfig< Topology, Topology::dimension, RealType, IndexType, LocalIndexType > >;

using MeshOfEdges = DefaultMeshTemplate< TNL::Meshes::Topologies::Edge >;
using MeshOfTriangles = DefaultMeshTemplate< TNL::Meshes::Topologies::Triangle >;
using MeshOfQuadrangles = DefaultMeshTemplate< TNL::Meshes::Topologies::Quadrangle >;
using MeshOfTetrahedrons = DefaultMeshTemplate< TNL::Meshes::Topologies::Tetrahedron >;
using MeshOfHexahedrons = DefaultMeshTemplate< TNL::Meshes::Topologies::Hexahedron >;
using MeshOfPolygons = DefaultMeshTemplate< TNL::Meshes::Topologies::Polygon >;
using MeshOfPolyhedrons = DefaultMeshTemplate< TNL::Meshes::Topologies::Polyhedron >;

using DistributedMeshOfEdges = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfEdges >;
using DistributedMeshOfTriangles = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfTriangles >;
using DistributedMeshOfQuadrangles = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfQuadrangles >;
using DistributedMeshOfTetrahedrons = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfTetrahedrons >;
using DistributedMeshOfHexahedrons = TNL::Meshes::DistributedMeshes::DistributedMesh< MeshOfHexahedrons >;

// Config tag for GridTypeResolver and MeshTypeResolver
struct PyTNLConfigTag
{};

namespace TNL::Meshes::BuildConfigTags {

// Note: cannot replace int with generic Index due to ambiguity :-(
template<>
struct GridRealTag< PyTNLConfigTag, float >
{
   static constexpr bool enabled = false;
};
template<>
struct GridRealTag< PyTNLConfigTag, RealType >
{
   static constexpr bool enabled = true;
};

// Note: cannot replace int with generic Index due to ambiguity :-(
template<>
struct GridIndexTag< PyTNLConfigTag, int >
{
   static constexpr bool enabled = false;
};
template<>
struct GridIndexTag< PyTNLConfigTag, IndexType >
{
   static constexpr bool enabled = true;
};

// Unstructured mesh topologies
template<>
struct MeshCellTopologyTag< PyTNLConfigTag, Topologies::Edge >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshCellTopologyTag< PyTNLConfigTag, Topologies::Triangle >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshCellTopologyTag< PyTNLConfigTag, Topologies::Quadrangle >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshCellTopologyTag< PyTNLConfigTag, Topologies::Polygon >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshCellTopologyTag< PyTNLConfigTag, Topologies::Tetrahedron >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshCellTopologyTag< PyTNLConfigTag, Topologies::Hexahedron >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshCellTopologyTag< PyTNLConfigTag, Topologies::Polyhedron >
{
   static constexpr bool enabled = true;
};

// Meshes are enabled only for the world dimension equal to the cell dimension.
template< typename CellTopology, int WorldDimension >
struct MeshSpaceDimensionTag< PyTNLConfigTag, CellTopology, WorldDimension >
{
   static constexpr bool enabled = WorldDimension == CellTopology::dimension;
};

// Meshes are enabled only for types explicitly listed below.
template<>
struct MeshRealTag< PyTNLConfigTag, RealType >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshGlobalIndexTag< PyTNLConfigTag, IndexType >
{
   static constexpr bool enabled = true;
};
template<>
struct MeshLocalIndexTag< PyTNLConfigTag, LocalIndexType >
{
   static constexpr bool enabled = true;
};

}  // namespace TNL::Meshes::BuildConfigTags

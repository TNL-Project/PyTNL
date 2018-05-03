#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Mesh.h>
#include <TNL/Meshes/DefaultConfig.h>
#include <TNL/Meshes/Topologies/Edge.h>
#include <TNL/Meshes/Topologies/Triangle.h>
#include <TNL/Meshes/Topologies/Tetrahedron.h>

using RealType = double;
using DeviceType = TNL::Devices::Host;
using IndexType = int;

using Grid1D = TNL::Meshes::Grid<1, RealType, DeviceType, IndexType>;
using Grid2D = TNL::Meshes::Grid<2, RealType, DeviceType, IndexType>;
using Grid3D = TNL::Meshes::Grid<3, RealType, DeviceType, IndexType>;

using LocalIndexType = short int;
using EdgeTopology = TNL::Meshes::Topologies::Edge;
using TriangleTopology = TNL::Meshes::Topologies::Triangle;
using TetrahedronTopology = TNL::Meshes::Topologies::Tetrahedron;
using MeshOfEdges = TNL::Meshes::Mesh< TNL::Meshes::DefaultConfig<
                            EdgeTopology,
                            EdgeTopology::dimension,
                            RealType,
                            IndexType,
                            LocalIndexType,
                            IndexType > >;
using MeshOfTriangles = TNL::Meshes::Mesh< TNL::Meshes::DefaultConfig<
                            TriangleTopology,
                            TriangleTopology::dimension,
                            RealType,
                            IndexType,
                            LocalIndexType,
                            IndexType > >;
using MeshOfTetrahedrons = TNL::Meshes::Mesh< TNL::Meshes::DefaultConfig<
                            TetrahedronTopology,
                            TetrahedronTopology::dimension,
                            RealType,
                            IndexType,
                            LocalIndexType,
                            IndexType > >;

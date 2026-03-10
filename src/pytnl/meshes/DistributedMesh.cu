#include "DistributedMesh.h"

void
export_DistributedMeshes( nb::module_& m )
{
   export_DistributedMesh< DistributedMeshOfEdges_cuda >( m, "DistributedMesh_Mesh_Edge" );
   export_DistributedMesh< DistributedMeshOfTriangles_cuda >( m, "DistributedMesh_Mesh_Triangle" );
   export_DistributedMesh< DistributedMeshOfQuadrangles_cuda >( m, "DistributedMesh_Mesh_Quadrangle" );
   export_DistributedMesh< DistributedMeshOfTetrahedrons_cuda >( m, "DistributedMesh_Mesh_Tetrahedron" );
   export_DistributedMesh< DistributedMeshOfHexahedrons_cuda >( m, "DistributedMesh_Mesh_Hexahedron" );
   export_DistributedMesh< DistributedMeshOfPolygons_cuda >( m, "DistributedMesh_Mesh_Polygon" );
   export_DistributedMesh< DistributedMeshOfPolyhedrons_cuda >( m, "DistributedMesh_Mesh_Polyhedron" );
}

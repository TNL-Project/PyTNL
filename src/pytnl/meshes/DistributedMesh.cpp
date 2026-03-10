#include <pytnl/pytnl.h>
#include <pytnl/containers/Array.h>

#include "DistributedMesh.h"

void
export_DistributedMeshes( nb::module_& m )
{
   export_DistributedMesh< DistributedMeshOfEdges_host >( m, "DistributedMesh_Mesh_Edge" );
   export_DistributedMesh< DistributedMeshOfTriangles_host >( m, "DistributedMesh_Mesh_Triangle" );
   export_DistributedMesh< DistributedMeshOfQuadrangles_host >( m, "DistributedMesh_Mesh_Quadrangle" );
   export_DistributedMesh< DistributedMeshOfTetrahedrons_host >( m, "DistributedMesh_Mesh_Tetrahedron" );
   export_DistributedMesh< DistributedMeshOfHexahedrons_host >( m, "DistributedMesh_Mesh_Hexahedron" );
   export_DistributedMesh< DistributedMeshOfPolygons_host >( m, "DistributedMesh_Mesh_Polygon" );
   export_DistributedMesh< DistributedMeshOfPolyhedrons_host >( m, "DistributedMesh_Mesh_Polyhedron" );

   // export VTKTypesArrayType
   using VTKTypesArrayType = typename DistributedMeshOfEdges_host::VTKTypesArrayType;
   export_Array< VTKTypesArrayType >( m, "VTKTypesArrayType" );
}

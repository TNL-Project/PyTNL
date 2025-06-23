#include <pytnl/pytnl.h>
#include <pytnl/containers/Array.h>

#include "DistributedMesh.h"

void
export_DistributedMeshes( nb::module_& m )
{
   export_DistributedMesh< DistributedMeshOfEdges >( m, "DistributedMeshOfEdges" );
   export_DistributedMesh< DistributedMeshOfTriangles >( m, "DistributedMeshOfTriangles" );
   export_DistributedMesh< DistributedMeshOfQuadrangles >( m, "DistributedMeshOfQuadrangles" );
   export_DistributedMesh< DistributedMeshOfTetrahedrons >( m, "DistributedMeshOfTetrahedrons" );
   export_DistributedMesh< DistributedMeshOfHexahedrons >( m, "DistributedMeshOfHexahedrons" );

   // export VTKTypesArrayType
   using VTKTypesArrayType = typename DistributedMeshOfEdges::VTKTypesArrayType;
   export_Array< VTKTypesArrayType >( m, "VTKTypesArrayType" );
}

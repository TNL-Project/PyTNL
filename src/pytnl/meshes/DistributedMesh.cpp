#include <pytnl/pytnl.h>
#include <pytnl/containers/Array.h>

#include "DistributedMesh.h"

void
export_DistributedMeshes( nb::module_& m )
{
   export_DistributedMesh< DistributedMeshOfEdges_host >( m, "DistributedMeshOfEdges" );
   export_DistributedMesh< DistributedMeshOfTriangles_host >( m, "DistributedMeshOfTriangles" );
   export_DistributedMesh< DistributedMeshOfQuadrangles_host >( m, "DistributedMeshOfQuadrangles" );
   export_DistributedMesh< DistributedMeshOfTetrahedrons_host >( m, "DistributedMeshOfTetrahedrons" );
   export_DistributedMesh< DistributedMeshOfHexahedrons_host >( m, "DistributedMeshOfHexahedrons" );

   // export VTKTypesArrayType
   using VTKTypesArrayType = typename DistributedMeshOfEdges_host::VTKTypesArrayType;
   export_Array< VTKTypesArrayType >( m, "VTKTypesArrayType" );
}

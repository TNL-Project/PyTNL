#include "DistributedMesh.h"

void
export_DistributedMeshes( nb::module_& m )
{
   export_DistributedMesh< DistributedMeshOfEdges_cuda >( m, "DistributedMeshOfEdges" );
   export_DistributedMesh< DistributedMeshOfTriangles_cuda >( m, "DistributedMeshOfTriangles" );
   export_DistributedMesh< DistributedMeshOfQuadrangles_cuda >( m, "DistributedMeshOfQuadrangles" );
   export_DistributedMesh< DistributedMeshOfTetrahedrons_cuda >( m, "DistributedMeshOfTetrahedrons" );
   export_DistributedMesh< DistributedMeshOfHexahedrons_cuda >( m, "DistributedMeshOfHexahedrons" );
}

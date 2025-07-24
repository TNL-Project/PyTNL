#include "Mesh.h"

void
export_Meshes( nb::module_& m )
{
   export_Mesh< MeshOfEdges_cuda >( m, "MeshOfEdges" );
   export_Mesh< MeshOfTriangles_cuda >( m, "MeshOfTriangles" );
   export_Mesh< MeshOfQuadrangles_cuda >( m, "MeshOfQuadrangles" );
   export_Mesh< MeshOfTetrahedrons_cuda >( m, "MeshOfTetrahedrons" );
   export_Mesh< MeshOfHexahedrons_cuda >( m, "MeshOfHexahedrons" );
   export_Mesh< MeshOfPolygons_cuda >( m, "MeshOfPolygons" );
   export_Mesh< MeshOfPolyhedrons_cuda >( m, "MeshOfPolyhedrons" );
}

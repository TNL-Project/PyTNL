#include "Mesh.h"

void
export_Meshes( nb::module_& m )
{
   export_Mesh< MeshOfEdges_host >( m, "MeshOfEdges" );
   export_Mesh< MeshOfTriangles_host >( m, "MeshOfTriangles" );
   export_Mesh< MeshOfQuadrangles_host >( m, "MeshOfQuadrangles" );
   export_Mesh< MeshOfTetrahedrons_host >( m, "MeshOfTetrahedrons" );
   export_Mesh< MeshOfHexahedrons_host >( m, "MeshOfHexahedrons" );
   export_Mesh< MeshOfPolygons_host >( m, "MeshOfPolygons" );
   export_Mesh< MeshOfPolyhedrons_host >( m, "MeshOfPolyhedrons" );
}

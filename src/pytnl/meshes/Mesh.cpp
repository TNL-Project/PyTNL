#include "Mesh.h"

void
export_Meshes( nb::module_& m )
{
   export_Mesh< MeshOfEdges >( m, "MeshOfEdges" );
   export_Mesh< MeshOfTriangles >( m, "MeshOfTriangles" );
   export_Mesh< MeshOfQuadrangles >( m, "MeshOfQuadrangles" );
   export_Mesh< MeshOfTetrahedrons >( m, "MeshOfTetrahedrons" );
   export_Mesh< MeshOfHexahedrons >( m, "MeshOfHexahedrons" );
   export_Mesh< MeshOfPolygons >( m, "MeshOfPolygons" );
   export_Mesh< MeshOfPolyhedrons >( m, "MeshOfPolyhedrons" );
}

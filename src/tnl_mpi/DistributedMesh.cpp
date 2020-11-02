// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "../typedefs.h"
#include "DistributedMesh.h"
#include "../tnl/Array.h"

void export_DistributedMeshes( py::module & m )
{
    // make sure that bindings for the local meshes are available
    py::module_::import("tnl");

    export_DistributedMesh< DistributedMeshOfEdges >( m, "DistributedMeshOfEdges" );
    export_DistributedMesh< DistributedMeshOfTriangles >( m, "DistributedMeshOfTriangles" );
    export_DistributedMesh< DistributedMeshOfTetrahedrons >( m, "DistributedMeshOfTetrahedrons" );

    // export VTKTypesArrayType
    using VTKTypesArrayType = typename DistributedMeshOfEdges::VTKTypesArrayType;
    export_Array< VTKTypesArrayType >(m, "VTKTypesArrayType");
}

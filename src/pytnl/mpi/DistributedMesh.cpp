// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include <pytnl/tnl/Array.h>
#include <pytnl/tnl_mpi/DistributedMesh.h>
#include <pytnl/typedefs.h>

void
export_DistributedMeshes( nb::module_& m )
{
   // make sure that bindings for the local meshes are available
   nb::module_::import_( "tnl" );

   export_DistributedMesh< DistributedMeshOfEdges >( m, "DistributedMeshOfEdges" );
   export_DistributedMesh< DistributedMeshOfTriangles >( m, "DistributedMeshOfTriangles" );
   export_DistributedMesh< DistributedMeshOfQuadrangles >( m, "DistributedMeshOfQuadrangles" );
   export_DistributedMesh< DistributedMeshOfTetrahedrons >( m, "DistributedMeshOfTetrahedrons" );
   export_DistributedMesh< DistributedMeshOfHexahedrons >( m, "DistributedMeshOfHexahedrons" );

   // export VTKTypesArrayType
   using VTKTypesArrayType = typename DistributedMeshOfEdges::VTKTypesArrayType;
   export_Array< VTKTypesArrayType >( m, "VTKTypesArrayType" );
}

#include <pytnl/pytnl.h>

#include <TNL/Meshes/DistributedMeshes/distributeSubentities.h>

void
export_distributeSubentities( nb::module_& m )
{
   using TNL::Meshes::DistributedMeshes::distributeSubentities;
   m.def( "distributeFaces",
          []( DistributedMeshOfTriangles& mesh )
          {
             distributeSubentities< 1 >( mesh );
          } );
   m.def( "distributeFaces",
          []( DistributedMeshOfQuadrangles& mesh )
          {
             distributeSubentities< 1 >( mesh );
          } );
   m.def( "distributeFaces",
          []( DistributedMeshOfTetrahedrons& mesh )
          {
             distributeSubentities< 2 >( mesh );
          } );
   m.def( "distributeFaces",
          []( DistributedMeshOfHexahedrons& mesh )
          {
             distributeSubentities< 2 >( mesh );
          } );
}

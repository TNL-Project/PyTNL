#include <pytnl/exceptions.h>
#include <pytnl/typedefs.h>

// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/MPI/Wrappers.h>

// external functions
void
export_DistributedMeshes( nb::module_& m );
void
export_DistributedMeshReaders( nb::module_& m );
void
export_DistributedMeshWriters( nb::module_& m );

#include <TNL/Meshes/DistributedMeshes/distributeSubentities.h>

// Python module definition
NB_MODULE( tnl_mpi, m )
{
   register_exceptions( m );

   // MPI initialization and finalization
   // https://stackoverflow.com/q/64647846
   if( ! TNL::MPI::Initialized() ) {
      int argc = 0;
      char** argv = nullptr;
      TNL::MPI::Init( argc, argv );
   }
   // https://pybind11.readthedocs.io/en/stable/advanced/misc.html#module-destructors
   auto cleanup_callback = []( void* ptr ) noexcept
   {
      if( TNL::MPI::Initialized() && ! TNL::MPI::Finalized() )
         TNL::MPI::Finalize();
   };
   m.attr( "_cleanup" ) = nb::capsule( nullptr, cleanup_callback );

   // bindings for distributed data structures
   export_DistributedMeshes( m );
   export_DistributedMeshReaders( m );
   export_DistributedMeshWriters( m );

   // bindings for functions
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

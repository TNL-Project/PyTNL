#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include <TNL/MPI/Utils.h>

void
export_ArrayVector( nb::module_& m );
void
export_NDArray( nb::module_& m );

// Python module definition
NB_MODULE( _containers_cuda, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers" );
   nb::module_::import_( "mpi4py.MPI" );

   // Importing mpi4py.MPI does MPI_Init, but it does not handle GPU selection.
   // Calling selectGPU() from the lowest-level GPU-accelerated module ensures
   // the same behavior as the TNL::MPI::Initialize wrapper function.
   if( TNL::MPI::isInitialized() )
      TNL::MPI::selectGPU();

   export_ArrayVector( m );
   export_NDArray( m );
}

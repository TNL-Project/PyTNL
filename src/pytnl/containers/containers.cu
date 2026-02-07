#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

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

   export_ArrayVector( m );
   export_NDArray( m );
}

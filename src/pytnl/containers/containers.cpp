#include <pytnl/exceptions.h>
#include <pytnl/typedefs.h>

// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include <pytnl/tnl/Array.h>
#include <pytnl/tnl/Vector.h>

template< typename T >
using _array = TNL::Containers::Array< T, TNL::Devices::Host, IndexType >;

template< typename T >
using _vector = TNL::Containers::Vector< T, TNL::Devices::Host, IndexType >;

// Python module definition
NB_MODULE( containers, m )
{
   register_exceptions( m );

   export_Array< _array< RealType > >( m, "Array" );
   export_Vector< _array< RealType >, _vector< RealType > >( m, "Vector" );
   export_Array< _array< IndexType > >( m, "Array_int" );
   export_Vector< _array< IndexType >, _vector< IndexType > >( m, "Vector_int" );
   export_Array< _array< bool > >( m, "Array_bool" );
}

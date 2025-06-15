#include <pytnl/exceptions.h>
#include <pytnl/typedefs.h>

// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include <pytnl/containers/Array.h>
#include <pytnl/containers/Vector.h>
#include <pytnl/containers/StaticVector.h>

using namespace TNL::Containers;

template< typename T >
using _array = TNL::Containers::Array< T, TNL::Devices::Host, IndexType >;

template< typename T >
using _vector = TNL::Containers::Vector< T, TNL::Devices::Host, IndexType >;

// Python module definition
NB_MODULE( _containers, m )
{
   register_exceptions( m );

   export_Array< _array< bool > >( m, "Array_bool" );
   export_Array< _array< IndexType > >( m, "Array_int" );
   export_Array< _array< RealType > >( m, "Array_float" );
   export_Vector< _array< IndexType >, _vector< IndexType > >( m, "Vector_int" );
   export_Vector< _array< RealType >, _vector< RealType > >( m, "Vector_float" );

   export_StaticVector< StaticVector< 1, IndexType > >( m, "StaticVector_1_int" );
   export_StaticVector< StaticVector< 2, IndexType > >( m, "StaticVector_2_int" );
   export_StaticVector< StaticVector< 3, IndexType > >( m, "StaticVector_3_int" );
   export_StaticVector< StaticVector< 1, RealType > >( m, "StaticVector_1_float" );
   export_StaticVector< StaticVector< 2, RealType > >( m, "StaticVector_2_float" );
   export_StaticVector< StaticVector< 3, RealType > >( m, "StaticVector_3_float" );
}

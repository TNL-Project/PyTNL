#include <pytnl/pytnl.h>

#include <pytnl/containers/DistributedArray.h>
#include <pytnl/containers/DistributedVector.h>

using namespace TNL::Containers;

template< typename T >
using _distributed_array = DistributedArray< T, TNL::Devices::Host, IndexType >;

template< typename T >
using _distributed_vector = DistributedVector< T, TNL::Devices::Host, IndexType >;

void
export_DistributedArrayVector( nb::module_& m )
{
   export_DistributedArray< _distributed_array< IndexType > >( m, "DistributedArray_int" );
   export_DistributedArray< _distributed_array< RealType > >( m, "DistributedArray_float" );
   export_DistributedArray< _distributed_array< ComplexType > >( m, "DistributedArray_complex" );

   export_DistributedVector< _distributed_array< IndexType >, _distributed_vector< IndexType > >( m, "DistributedVector_int" );
   export_DistributedVector< _distributed_array< RealType >, _distributed_vector< RealType > >( m, "DistributedVector_float" );
   export_DistributedVector< _distributed_array< ComplexType >, _distributed_vector< ComplexType > >(
      m, "DistributedVector_complex" );
}

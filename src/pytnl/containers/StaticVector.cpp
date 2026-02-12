#include <pytnl/pytnl.h>

#include <pytnl/containers/StaticVector.h>

using namespace TNL::Containers;

void
export_StaticVector( nb::module_& m )
{
   export_StaticVector< StaticVector< 1, IndexType > >( m, "StaticVector_1_int" );
   export_StaticVector< StaticVector< 2, IndexType > >( m, "StaticVector_2_int" );
   export_StaticVector< StaticVector< 3, IndexType > >( m, "StaticVector_3_int" );
   export_StaticVector< StaticVector< 1, RealType > >( m, "StaticVector_1_float" );
   export_StaticVector< StaticVector< 2, RealType > >( m, "StaticVector_2_float" );
   export_StaticVector< StaticVector< 3, RealType > >( m, "StaticVector_3_float" );
   export_StaticVector< StaticVector< 1, ComplexType > >( m, "StaticVector_1_complex" );
   export_StaticVector< StaticVector< 2, ComplexType > >( m, "StaticVector_2_complex" );
   export_StaticVector< StaticVector< 3, ComplexType > >( m, "StaticVector_3_complex" );
}

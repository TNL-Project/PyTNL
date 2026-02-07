#include <pytnl/pytnl.h>

#include <pytnl/containers/Array.h>
#include <pytnl/containers/Vector.h>
#include <pytnl/complex_caster.h>
#include <TNL/Arithmetics/Complex.h>

using namespace TNL::Containers;

template< typename T >
using _array = Array< T, TNL::Devices::Cuda, IndexType >;

template< typename T >
using _vector = Vector< T, TNL::Devices::Cuda, IndexType >;

template< typename T >
using _array_view = ArrayView< T, TNL::Devices::Cuda, IndexType >;

template< typename T >
using _vector_view = VectorView< T, TNL::Devices::Cuda, IndexType >;

void
export_ArrayVector( nb::module_& m )
{
   // std::complex does not work with CUDA (even in C++20)
   using ComplexType = TNL::Arithmetics::Complex< RealType >;

   export_Array< _array< bool > >( m, "Array_bool" );
   export_Array< _array< IndexType > >( m, "Array_int" );
   export_Array< _array< RealType > >( m, "Array_float" );
   export_Array< _array< ComplexType > >( m, "Array_complex" );
   export_Vector< _array< IndexType >, _vector< IndexType > >( m, "Vector_int" );
   export_Vector< _array< RealType >, _vector< RealType > >( m, "Vector_float" );
   export_Vector< _array< ComplexType >, _vector< ComplexType > >( m, "Vector_complex" );

   export_Array< _array_view< bool > >( m, "ArrayView_bool" );
   export_Array< _array_view< IndexType > >( m, "ArrayView_int" );
   export_Array< _array_view< RealType > >( m, "ArrayView_float" );
   export_Array< _array_view< ComplexType > >( m, "ArrayView_complex" );
   export_Vector< _array_view< IndexType >, _vector_view< IndexType > >( m, "VectorView_int" );
   export_Vector< _array_view< RealType >, _vector_view< RealType > >( m, "VectorView_float" );
   export_Vector< _array_view< ComplexType >, _vector_view< ComplexType > >( m, "VectorView_complex" );

   export_Array< _array_view< bool const > >( m, "ArrayView_bool_const" );
   export_Array< _array_view< IndexType const > >( m, "ArrayView_int_const" );
   export_Array< _array_view< RealType const > >( m, "ArrayView_float_const" );
   export_Array< _array_view< ComplexType const > >( m, "ArrayView_complex_const" );
   export_Vector< _array_view< IndexType const >, _vector_view< IndexType const > >( m, "VectorView_int_const" );
   export_Vector< _array_view< RealType const >, _vector_view< RealType const > >( m, "VectorView_float_const" );
   export_Vector< _array_view< ComplexType const >, _vector_view< ComplexType const > >( m, "VectorView_complex_const" );
}

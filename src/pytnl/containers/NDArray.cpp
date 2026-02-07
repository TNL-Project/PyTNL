#include <pytnl/pytnl.h>

#include <pytnl/containers/NDArray.h>

using namespace TNL::Containers;

template< std::size_t dim >
using _ndindexer =
   NDArrayIndexer< make_sizes_holder< IndexType, dim >,  // all sizes are set at runtime
                   make_sizes_holder< IndexType, dim >,  // all strides are set at runtime
                   make_sizes_holder< IndexType, dim >   // all overlaps are set at runtime
                   //ConstStaticSizesHolder< IndexType, dim, 0 >  // ConstStaticSizesHolder does not have Python bindings
                   >;

template< int dim, typename T >
using _ndarray = NDArray< T,
                          make_sizes_holder< IndexType, dim >,
                          std::make_index_sequence< dim >,  // identity by default
                          TNL::Devices::Host,
                          IndexType,
                          make_sizes_holder< IndexType, dim >  // all overlaps are set at runtime
                          //ConstStaticSizesHolder< IndexType, dim, 0 >  // ConstStaticSizesHolder does not have Python bindings
                          >;

template< int dim, typename T >
using _ndarray_view = typename _ndarray< dim, T >::ViewType;

void
export_NDArray( nb::module_& m )
{
   export_NDArrayIndexer< _ndindexer< 1 > >( m, "NDArrayIndexer_1" );
   export_NDArrayIndexer< _ndindexer< 2 > >( m, "NDArrayIndexer_2" );
   export_NDArrayIndexer< _ndindexer< 3 > >( m, "NDArrayIndexer_3" );

   export_NDArray< _ndarray< 1, IndexType > >( m, "NDArray_1_int" );
   export_NDArray< _ndarray< 2, IndexType > >( m, "NDArray_2_int" );
   export_NDArray< _ndarray< 3, IndexType > >( m, "NDArray_3_int" );
   export_NDArray< _ndarray< 1, RealType > >( m, "NDArray_1_float" );
   export_NDArray< _ndarray< 2, RealType > >( m, "NDArray_2_float" );
   export_NDArray< _ndarray< 3, RealType > >( m, "NDArray_3_float" );
   export_NDArray< _ndarray< 1, ComplexType > >( m, "NDArray_1_complex" );
   export_NDArray< _ndarray< 2, ComplexType > >( m, "NDArray_2_complex" );
   export_NDArray< _ndarray< 3, ComplexType > >( m, "NDArray_3_complex" );

   export_NDArray< _ndarray_view< 1, IndexType > >( m, "NDArrayView_1_int" );
   export_NDArray< _ndarray_view< 2, IndexType > >( m, "NDArrayView_2_int" );
   export_NDArray< _ndarray_view< 3, IndexType > >( m, "NDArrayView_3_int" );
   export_NDArray< _ndarray_view< 1, RealType > >( m, "NDArrayView_1_float" );
   export_NDArray< _ndarray_view< 2, RealType > >( m, "NDArrayView_2_float" );
   export_NDArray< _ndarray_view< 3, RealType > >( m, "NDArrayView_3_float" );
   export_NDArray< _ndarray_view< 1, ComplexType > >( m, "NDArrayView_1_complex" );
   export_NDArray< _ndarray_view< 2, ComplexType > >( m, "NDArrayView_2_complex" );
   export_NDArray< _ndarray_view< 3, ComplexType > >( m, "NDArrayView_3_complex" );

   export_NDArray< _ndarray_view< 1, IndexType const > >( m, "NDArrayView_1_int_const" );
   export_NDArray< _ndarray_view< 2, IndexType const > >( m, "NDArrayView_2_int_const" );
   export_NDArray< _ndarray_view< 3, IndexType const > >( m, "NDArrayView_3_int_const" );
   export_NDArray< _ndarray_view< 1, RealType const > >( m, "NDArrayView_1_float_const" );
   export_NDArray< _ndarray_view< 2, RealType const > >( m, "NDArrayView_2_float_const" );
   export_NDArray< _ndarray_view< 3, RealType const > >( m, "NDArrayView_3_float_const" );
   export_NDArray< _ndarray_view< 1, ComplexType const > >( m, "NDArrayView_1_complex_const" );
   export_NDArray< _ndarray_view< 2, ComplexType const > >( m, "NDArrayView_2_complex_const" );
   export_NDArray< _ndarray_view< 3, ComplexType const > >( m, "NDArrayView_3_complex_const" );
}

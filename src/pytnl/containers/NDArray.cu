#include <pytnl/pytnl.h>

#include <pytnl/containers/NDArray.h>
#include <pytnl/containers/DistributedNDArray.h>
#include <pytnl/complex_caster.h>
#include <TNL/Arithmetics/Complex.h>

using namespace TNL::Containers;

template< int dim, typename T >
using _ndarray = NDArray< T,
                          make_sizes_holder< IndexType, dim >,
                          std::make_index_sequence< dim >,  // identity by default
                          TNL::Devices::Cuda,
                          IndexType,
                          make_sizes_holder< IndexType, dim >  // all overlaps are set at runtime
                          //ConstStaticSizesHolder< IndexType, dim, 0 >  // ConstStaticSizesHolder does not have Python bindings
                          >;

template< int dim, typename T >
using _ndarray_view = typename _ndarray< dim, T >::ViewType;

template< int dim, typename T >
using _distributed_ndarray = DistributedNDArray< _ndarray< dim, T > >;

template< int dim, typename T >
using _distributed_ndarray_view = typename _distributed_ndarray< dim, T >::ViewType;

void
export_NDArray( nb::module_& m )
{
   // std::complex does not work with CUDA (even in C++20)
   using ComplexType = TNL::Arithmetics::Complex< RealType >;

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

   export_DistributedNDArray< _distributed_ndarray< 1, IndexType > >( m, "DistributedNDArray_1_int" );
   export_DistributedNDArray< _distributed_ndarray< 2, IndexType > >( m, "DistributedNDArray_2_int" );
   export_DistributedNDArray< _distributed_ndarray< 3, IndexType > >( m, "DistributedNDArray_3_int" );
   export_DistributedNDArray< _distributed_ndarray< 1, RealType > >( m, "DistributedNDArray_1_float" );
   export_DistributedNDArray< _distributed_ndarray< 2, RealType > >( m, "DistributedNDArray_2_float" );
   export_DistributedNDArray< _distributed_ndarray< 3, RealType > >( m, "DistributedNDArray_3_float" );
   export_DistributedNDArray< _distributed_ndarray< 1, ComplexType > >( m, "DistributedNDArray_1_complex" );
   export_DistributedNDArray< _distributed_ndarray< 2, ComplexType > >( m, "DistributedNDArray_2_complex" );
   export_DistributedNDArray< _distributed_ndarray< 3, ComplexType > >( m, "DistributedNDArray_3_complex" );

   export_DistributedNDArray< _distributed_ndarray_view< 1, IndexType > >( m, "DistributedNDArrayView_1_int" );
   export_DistributedNDArray< _distributed_ndarray_view< 2, IndexType > >( m, "DistributedNDArrayView_2_int" );
   export_DistributedNDArray< _distributed_ndarray_view< 3, IndexType > >( m, "DistributedNDArrayView_3_int" );
   export_DistributedNDArray< _distributed_ndarray_view< 1, RealType > >( m, "DistributedNDArrayView_1_float" );
   export_DistributedNDArray< _distributed_ndarray_view< 2, RealType > >( m, "DistributedNDArrayView_2_float" );
   export_DistributedNDArray< _distributed_ndarray_view< 3, RealType > >( m, "DistributedNDArrayView_3_float" );
   export_DistributedNDArray< _distributed_ndarray_view< 1, ComplexType > >( m, "DistributedNDArrayView_1_complex" );
   export_DistributedNDArray< _distributed_ndarray_view< 2, ComplexType > >( m, "DistributedNDArrayView_2_complex" );
   export_DistributedNDArray< _distributed_ndarray_view< 3, ComplexType > >( m, "DistributedNDArrayView_3_complex" );

   export_DistributedNDArray< _distributed_ndarray_view< 1, IndexType const > >( m, "DistributedNDArrayView_1_int_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 2, IndexType const > >( m, "DistributedNDArrayView_2_int_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 3, IndexType const > >( m, "DistributedNDArrayView_3_int_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 1, RealType const > >( m, "DistributedNDArrayView_1_float_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 2, RealType const > >( m, "DistributedNDArrayView_2_float_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 3, RealType const > >( m, "DistributedNDArrayView_3_float_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 1, ComplexType const > >( m,
                                                                                   "DistributedNDArrayView_1_complex_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 2, ComplexType const > >( m,
                                                                                   "DistributedNDArrayView_2_complex_const" );
   export_DistributedNDArray< _distributed_ndarray_view< 3, ComplexType const > >( m,
                                                                                   "DistributedNDArrayView_3_complex_const" );
}

#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Arithmetics/Complex.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseOperations.h>

#include <pytnl/complex_caster.h>

#include "DenseMatrix.h"
#include "SparseMatrix.h"

template< typename Device, typename Index, typename IndexAllocator >
using CSR = TNL::Algorithms::Segments::CSR< Device, Index, IndexAllocator >;
template< typename Device, typename Index, typename IndexAllocator >
using Ellpack = TNL::Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;
template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpack = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

// Sparse - Cuda
template< typename T >
using Sparse_CSR_cuda = TNL::Matrices::SparseMatrix< T, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, CSR >;
template< typename T >
using Sparse_Ell_cuda = TNL::Matrices::SparseMatrix< T, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, Ellpack >;
template< typename T >
using Sparse_SE_cuda =
   TNL::Matrices::SparseMatrix< T, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, SlicedEllpack >;

// Sparse - Host (needed for cross-device copies)
template< typename T >
using Sparse_CSR_host = TNL::Matrices::SparseMatrix< T, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, CSR >;
template< typename T >
using Sparse_Ell_host = TNL::Matrices::SparseMatrix< T, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, Ellpack >;
template< typename T >
using Sparse_SE_host =
   TNL::Matrices::SparseMatrix< T, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, SlicedEllpack >;

// Sparse base - Cuda
template< typename T >
using SparseBase_CSR_cuda = TNL::Matrices::SparseMatrixBase<
   T,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_CSR_cuda< T >::SegmentsType::ViewType,
   T >;
template< typename T >
using SparseBase_Ell_cuda = TNL::Matrices::SparseMatrixBase<
   T,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_Ell_cuda< T >::SegmentsType::ViewType,
   T >;
template< typename T >
using SparseBase_SE_cuda = TNL::Matrices::SparseMatrixBase<
   T,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_SE_cuda< T >::SegmentsType::ViewType,
   T >;

// Sparse const base types - cannot use std::add_const_t<T> here because the
// segments parameter must also change from ViewType to ConstViewType.
template< typename T >
using SparseBase_CSR_cuda_const = TNL::Matrices::SparseMatrixBase<
   const T,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_CSR_cuda< T >::SegmentsType::ViewType::ConstViewType,
   T >;
template< typename T >
using SparseBase_Ell_cuda_const = TNL::Matrices::SparseMatrixBase<
   const T,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_Ell_cuda< T >::SegmentsType::ViewType::ConstViewType,
   T >;
template< typename T >
using SparseBase_SE_cuda_const = TNL::Matrices::SparseMatrixBase<
   const T,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_SE_cuda< T >::SegmentsType::ViewType::ConstViewType,
   T >;

// Dense - Cuda
template< typename T >
using Dense_RowMajor_cuda =
   TNL::Matrices::DenseMatrix< T, TNL::Devices::Cuda, IndexType, TNL::Algorithms::Segments::RowMajorOrder >;
template< typename T >
using Dense_ColumnMajor_cuda =
   TNL::Matrices::DenseMatrix< T, TNL::Devices::Cuda, IndexType, TNL::Algorithms::Segments::ColumnMajorOrder >;

// Dense base - Cuda
template< typename T >
using DenseBase_RowMajor_cuda =
   TNL::Matrices::DenseMatrixBase< T, TNL::Devices::Cuda, IndexType, TNL::Algorithms::Segments::RowMajorOrder >;
template< typename T >
using DenseBase_ColumnMajor_cuda =
   TNL::Matrices::DenseMatrixBase< T, TNL::Devices::Cuda, IndexType, TNL::Algorithms::Segments::ColumnMajorOrder >;

// Segments types don't depend on value type, only on format. Export them once
// at the module level with format-specific names.
void
export_segments_types( nb::module_& m )
{
   export_Segments< typename Sparse_CSR_cuda< RealType >::SegmentsType >( m, "Segments_CSR" );
   export_Segments< typename Sparse_Ell_cuda< RealType >::SegmentsType >( m, "Segments_Ellpack" );
   export_Segments< typename Sparse_SE_cuda< RealType >::SegmentsType >( m, "Segments_SlicedEllpack" );
}

// Exports all CUDA matrix bindings (base classes, matrices, views, row views,
// and same-device copies) for a single value type T.
template< typename T >
void
export_value_type( nb::module_& m, const char* vn )
{
   const std::string valuename = vn;
   const std::string csr = valuename + "_CSR";
   const std::string e = valuename + "_Ellpack";
   const std::string se = valuename + "_SlicedEllpack";
   const std::string row = valuename + "_RowMajor";
   const std::string col = valuename + "_ColumnMajor";

   // Base classes
   export_SparseMatrixBaseClass< SparseBase_CSR_cuda< T > >( m, ( "SparseMatrixBase_" + csr ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_Ell_cuda< T > >( m, ( "SparseMatrixBase_" + e ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_SE_cuda< T > >( m, ( "SparseMatrixBase_" + se ).c_str() );

   export_SparseMatrixBaseClass< SparseBase_CSR_cuda_const< T > >( m, ( "SparseMatrixBase_" + csr + "_const" ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_Ell_cuda_const< T > >( m, ( "SparseMatrixBase_" + e + "_const" ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_SE_cuda_const< T > >( m, ( "SparseMatrixBase_" + se + "_const" ).c_str() );

   export_DenseMatrixBaseClass< DenseBase_RowMajor_cuda< T > >( m, ( "DenseMatrixBase_" + row ).c_str() );
   export_DenseMatrixBaseClass< DenseBase_RowMajor_cuda< std::add_const_t< T > > >(
      m, ( "DenseMatrixBase_" + row + "_const" ).c_str() );
   export_DenseMatrixBaseClass< DenseBase_ColumnMajor_cuda< T > >( m, ( "DenseMatrixBase_" + col ).c_str() );
   export_DenseMatrixBaseClass< DenseBase_ColumnMajor_cuda< std::add_const_t< T > > >(
      m, ( "DenseMatrixBase_" + col + "_const" ).c_str() );

   // Sparse matrices
   export_Matrix< Sparse_CSR_cuda< T >, SparseBase_CSR_cuda< T > >( m, ( "SparseMatrix_" + csr ).c_str() );
   export_Matrix< Sparse_Ell_cuda< T >, SparseBase_Ell_cuda< T > >( m, ( "SparseMatrix_" + e ).c_str() );
   export_Matrix< Sparse_SE_cuda< T >, SparseBase_SE_cuda< T > >( m, ( "SparseMatrix_" + se ).c_str() );

   export_RowView< typename Sparse_CSR_cuda< T >::RowView >( m, ( "SparseMatrixRowView_" + valuename ).c_str() );
   export_RowView< typename Sparse_CSR_cuda< T >::ConstRowView >( m, ( "SparseMatrixConstRowView_" + valuename ).c_str() );

   // Sparse matrix views
   export_SparseMatrixView< typename Sparse_CSR_cuda< T >::ViewType, SparseBase_CSR_cuda< T > >(
      m, ( "SparseMatrixView_" + csr ).c_str() );
   export_SparseMatrixView< typename Sparse_Ell_cuda< T >::ViewType, SparseBase_Ell_cuda< T > >(
      m, ( "SparseMatrixView_" + e ).c_str() );
   export_SparseMatrixView< typename Sparse_SE_cuda< T >::ViewType, SparseBase_SE_cuda< T > >(
      m, ( "SparseMatrixView_" + se ).c_str() );
   export_SparseMatrixView< typename Sparse_CSR_cuda< T >::ConstViewType, SparseBase_CSR_cuda_const< T > >(
      m, ( "SparseMatrixView_" + csr + "_const" ).c_str() );
   export_SparseMatrixView< typename Sparse_Ell_cuda< T >::ConstViewType, SparseBase_Ell_cuda_const< T > >(
      m, ( "SparseMatrixView_" + e + "_const" ).c_str() );
   export_SparseMatrixView< typename Sparse_SE_cuda< T >::ConstViewType, SparseBase_SE_cuda_const< T > >(
      m, ( "SparseMatrixView_" + se + "_const" ).c_str() );

   // Same-device (Cuda) copies
   def_copySparseMatrix< Sparse_CSR_cuda< T >, Sparse_Ell_cuda< T > >( m );
   def_copySparseMatrix< Sparse_Ell_cuda< T >, Sparse_CSR_cuda< T > >( m );
   def_copySparseMatrix< Sparse_CSR_cuda< T >, Sparse_SE_cuda< T > >( m );
   def_copySparseMatrix< Sparse_SE_cuda< T >, Sparse_CSR_cuda< T > >( m );
   def_copySparseMatrix< Sparse_Ell_cuda< T >, Sparse_SE_cuda< T > >( m );
   def_copySparseMatrix< Sparse_SE_cuda< T >, Sparse_Ell_cuda< T > >( m );

   // Cross-device copies require both matrices to have the same RealType
   // (TNL::Matrices::copySparseMatrix static_asserts on this).
   // For float, RealType is `double` on both Host and Cuda.
   // For complex, Host uses std::complex<double> and Cuda uses TNL::Arithmetics::Complex<double>,
   // which are distinct types - so cross-device complex copies are not available.
   if constexpr( std::is_same_v< T, RealType > ) {
      // destination=Host, source=Cuda
      def_copySparseMatrix< Sparse_CSR_host< T >, Sparse_CSR_cuda< T > >( m );
      def_copySparseMatrix< Sparse_CSR_host< T >, Sparse_Ell_cuda< T > >( m );
      def_copySparseMatrix< Sparse_CSR_host< T >, Sparse_SE_cuda< T > >( m );
      def_copySparseMatrix< Sparse_Ell_host< T >, Sparse_CSR_cuda< T > >( m );
      def_copySparseMatrix< Sparse_Ell_host< T >, Sparse_Ell_cuda< T > >( m );
      def_copySparseMatrix< Sparse_Ell_host< T >, Sparse_SE_cuda< T > >( m );
      def_copySparseMatrix< Sparse_SE_host< T >, Sparse_CSR_cuda< T > >( m );
      def_copySparseMatrix< Sparse_SE_host< T >, Sparse_Ell_cuda< T > >( m );
      def_copySparseMatrix< Sparse_SE_host< T >, Sparse_SE_cuda< T > >( m );

      // destination=Cuda, source=Host
      def_copySparseMatrix< Sparse_CSR_cuda< T >, Sparse_CSR_host< T > >( m );
      def_copySparseMatrix< Sparse_CSR_cuda< T >, Sparse_Ell_host< T > >( m );
      def_copySparseMatrix< Sparse_CSR_cuda< T >, Sparse_SE_host< T > >( m );
      def_copySparseMatrix< Sparse_Ell_cuda< T >, Sparse_CSR_host< T > >( m );
      def_copySparseMatrix< Sparse_Ell_cuda< T >, Sparse_Ell_host< T > >( m );
      def_copySparseMatrix< Sparse_Ell_cuda< T >, Sparse_SE_host< T > >( m );
      def_copySparseMatrix< Sparse_SE_cuda< T >, Sparse_CSR_host< T > >( m );
      def_copySparseMatrix< Sparse_SE_cuda< T >, Sparse_Ell_host< T > >( m );
      def_copySparseMatrix< Sparse_SE_cuda< T >, Sparse_SE_host< T > >( m );
   }

   // Dense matrices
   export_DenseMatrix< Dense_ColumnMajor_cuda< T >, DenseBase_ColumnMajor_cuda< T > >( m, ( "DenseMatrix_" + col ).c_str() );
   export_DenseMatrix< Dense_RowMajor_cuda< T >, DenseBase_RowMajor_cuda< T > >( m, ( "DenseMatrix_" + row ).c_str() );

   export_DenseRowView< typename Dense_ColumnMajor_cuda< T >::RowView >( m, ( "DenseMatrixRowView_" + col ).c_str() );
   export_DenseRowView< typename Dense_ColumnMajor_cuda< T >::ConstRowView >( m, ( "DenseMatrixConstRowView_" + col ).c_str() );
   export_DenseRowView< typename Dense_RowMajor_cuda< T >::RowView >( m, ( "DenseMatrixRowView_" + row ).c_str() );
   export_DenseRowView< typename Dense_RowMajor_cuda< T >::ConstRowView >( m, ( "DenseMatrixConstRowView_" + row ).c_str() );

   export_DenseMatrixView< typename Dense_ColumnMajor_cuda< T >::ViewType, DenseBase_ColumnMajor_cuda< T > >(
      m, ( "DenseMatrixView_" + col ).c_str() );
   export_DenseMatrixView<
      typename Dense_ColumnMajor_cuda< T >::ConstViewType,
      DenseBase_ColumnMajor_cuda< std::add_const_t< T > > >( m, ( "DenseMatrixView_" + col + "_const" ).c_str() );
   export_DenseMatrixView< typename Dense_RowMajor_cuda< T >::ViewType, DenseBase_RowMajor_cuda< T > >(
      m, ( "DenseMatrixView_" + row ).c_str() );
   export_DenseMatrixView< typename Dense_RowMajor_cuda< T >::ConstViewType, DenseBase_RowMajor_cuda< std::add_const_t< T > > >(
      m, ( "DenseMatrixView_" + row + "_const" ).c_str() );
}

// Python module definition
NB_MODULE( _matrices_cuda, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers_cuda" );
   // Import the Host module so the ElementsOrganization enum type caster
   // is available for getOrganization() on CUDA matrices.
   nb::module_::import_( "pytnl._matrices" );

   // Format tags and ElementsOrganization are not exported here — they are
   // device-independent and defined only in the Host module.

   export_segments_types( m );

   // On CUDA, std::complex does not work - use TNL::Arithmetics::Complex.
   using CudaComplex = TNL::Arithmetics::Complex< RealType >;
   export_value_type< RealType >( m, "float" );
   export_value_type< CudaComplex >( m, "complex" );
}

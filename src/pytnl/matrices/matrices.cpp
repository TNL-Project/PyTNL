#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseOperations.h>

#include "DenseMatrix.h"
#include "SparseMatrix.h"

template< typename Device, typename Index, typename IndexAllocator >
using CSR = TNL::Algorithms::Segments::CSR< Device, Index, IndexAllocator >;
template< typename Device, typename Index, typename IndexAllocator >
using Ellpack = TNL::Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;
template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpack = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

// Sparse matrix types
template< typename T >
using Sparse_CSR = TNL::Matrices::SparseMatrix< T, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, CSR >;
template< typename T >
using Sparse_Ell = TNL::Matrices::SparseMatrix< T, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, Ellpack >;
template< typename T >
using Sparse_SE = TNL::Matrices::SparseMatrix< T, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, SlicedEllpack >;

// Sparse base types
template< typename T >
using SparseBase_CSR = TNL::Matrices::SparseMatrixBase<
   T,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_CSR< T >::SegmentsType::ViewType,
   T >;
template< typename T >
using SparseBase_Ell = TNL::Matrices::SparseMatrixBase<
   T,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_Ell< T >::SegmentsType::ViewType,
   T >;
template< typename T >
using SparseBase_SE = TNL::Matrices::SparseMatrixBase<
   T,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_SE< T >::SegmentsType::ViewType,
   T >;

// Sparse const base types - cannot use std::add_const_t<T> here because the
// segments parameter must also change from ViewType to ConstViewType.
template< typename T >
using SparseBase_CSR_const = TNL::Matrices::SparseMatrixBase<
   const T,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_CSR< T >::SegmentsType::ViewType::ConstViewType,
   T >;
template< typename T >
using SparseBase_Ell_const = TNL::Matrices::SparseMatrixBase<
   const T,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_Ell< T >::SegmentsType::ViewType::ConstViewType,
   T >;
template< typename T >
using SparseBase_SE_const = TNL::Matrices::SparseMatrixBase<
   const T,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename Sparse_SE< T >::SegmentsType::ViewType::ConstViewType,
   T >;

// Dense matrix types
template< typename T >
using Dense_RowMajor = TNL::Matrices::DenseMatrix< T, TNL::Devices::Host, IndexType, TNL::Algorithms::Segments::RowMajorOrder >;
template< typename T >
using Dense_ColumnMajor =
   TNL::Matrices::DenseMatrix< T, TNL::Devices::Host, IndexType, TNL::Algorithms::Segments::ColumnMajorOrder >;

// Dense base types
template< typename T >
using DenseBase_RowMajor =
   TNL::Matrices::DenseMatrixBase< T, TNL::Devices::Host, IndexType, TNL::Algorithms::Segments::RowMajorOrder >;
template< typename T >
using DenseBase_ColumnMajor =
   TNL::Matrices::DenseMatrixBase< T, TNL::Devices::Host, IndexType, TNL::Algorithms::Segments::ColumnMajorOrder >;

void
export_format_tags( nb::module_& m )
{
   auto submodule = m.def_submodule( "formats" );

   // NOTE: The SparseMatrix class template in C++ has a template-template
   // parameter of this form:
   //    template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments = Algorithms::Segments::CSR,
   // As this is not usable in Python bindings, we add bindings for empty tag
   // classes instead to facilitate format selection from Python. Each tag
   // class is combined with a Device and Index types in Python and mapped to
   // the appropriate Segments class.
   struct CSR
   {};
   struct Ellpack
   {};
   struct SlicedEllpack
   {};

   nb::class_< CSR >( submodule, "CSR", "Compressed Sparse Row format" );
   nb::class_< Ellpack >( submodule, "Ellpack", "ELLPACK format" );
   nb::class_< SlicedEllpack >( submodule, "SlicedEllpack", "Sliced ELLPACK format" );
}

void
export_organizations( nb::module_& m )
{
   // Bind the TNL ElementsOrganization enum so getOrganization() can return
   // a proper Python enum value instead of an unbound C++ integer. Also used
   // as the organization parameter in DenseMatrix[float, Host, ElementsOrganization.RowMajorOrder].
   nb::enum_< TNL::Algorithms::Segments::ElementsOrganization >( m, "ElementsOrganization" )
      .value( "ColumnMajorOrder", TNL::Algorithms::Segments::ColumnMajorOrder )
      .value( "RowMajorOrder", TNL::Algorithms::Segments::RowMajorOrder );
}

// Segments types don't depend on value type, only on format.
// Export them once at the module level with format-specific names.
void
export_segments_types( nb::module_& m )
{
   export_Segments< typename Sparse_CSR< RealType >::SegmentsType >( m, "Segments_CSR" );
   export_Segments< typename Sparse_Ell< RealType >::SegmentsType >( m, "Segments_Ellpack" );
   export_Segments< typename Sparse_SE< RealType >::SegmentsType >( m, "Segments_SlicedEllpack" );
}

// Exports all matrix bindings (base classes, matrices, views, row views, and same-device copy functions)
// for a single value type T.
// The value name (e.g. "float", "complex") is used to construct the Python-visible class names.
template< typename T >
void
export_value_type( nb::module_& m, const char* vn )
{
   const std::string valuename = vn;
   const std::string csr = valuename + "_CSR";
   const std::string ellpack = valuename + "_Ellpack";
   const std::string slicedellpack = valuename + "_SlicedEllpack";
   const std::string rowmajor = valuename + "_RowMajor";
   const std::string columnmajor = valuename + "_ColumnMajor";

   // Base classes
   export_SparseMatrixBaseClass< SparseBase_CSR< T > >( m, ( "SparseMatrixBase_" + csr ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_Ell< T > >( m, ( "SparseMatrixBase_" + ellpack ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_SE< T > >( m, ( "SparseMatrixBase_" + slicedellpack ).c_str() );

   export_SparseMatrixBaseClass< SparseBase_CSR_const< T > >( m, ( "SparseMatrixBase_" + csr + "_const" ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_Ell_const< T > >( m, ( "SparseMatrixBase_" + ellpack + "_const" ).c_str() );
   export_SparseMatrixBaseClass< SparseBase_SE_const< T > >( m, ( "SparseMatrixBase_" + slicedellpack + "_const" ).c_str() );

   export_DenseMatrixBaseClass< DenseBase_RowMajor< T > >( m, ( "DenseMatrixBase_" + rowmajor ).c_str() );
   export_DenseMatrixBaseClass< DenseBase_RowMajor< std::add_const_t< T > > >(
      m, ( "DenseMatrixBase_" + rowmajor + "_const" ).c_str() );
   export_DenseMatrixBaseClass< DenseBase_ColumnMajor< T > >( m, ( "DenseMatrixBase_" + columnmajor ).c_str() );
   export_DenseMatrixBaseClass< DenseBase_ColumnMajor< std::add_const_t< T > > >(
      m, ( "DenseMatrixBase_" + columnmajor + "_const" ).c_str() );

   // Sparse matrices
   export_Matrix< Sparse_CSR< T >, SparseBase_CSR< T > >( m, ( "SparseMatrix_" + csr ).c_str() );
   export_Matrix< Sparse_Ell< T >, SparseBase_Ell< T > >( m, ( "SparseMatrix_" + ellpack ).c_str() );
   export_Matrix< Sparse_SE< T >, SparseBase_SE< T > >( m, ( "SparseMatrix_" + slicedellpack ).c_str() );

   // NOTE: all exported formats (CSR, Ellpack, SlicedEllpack) use the same
   // SegmentView, so the RowView and ConstRowView are also the same types in
   // all three formats - but they differ per value type.
   export_RowView< typename Sparse_CSR< T >::RowView >( m, ( "SparseMatrixRowView_" + valuename ).c_str() );
   export_RowView< typename Sparse_CSR< T >::ConstRowView >( m, ( "SparseMatrixConstRowView_" + valuename ).c_str() );

   // Sparse matrix views
   export_SparseMatrixView< typename Sparse_CSR< T >::ViewType, SparseBase_CSR< T > >(
      m, ( "SparseMatrixView_" + csr ).c_str() );
   export_SparseMatrixView< typename Sparse_Ell< T >::ViewType, SparseBase_Ell< T > >(
      m, ( "SparseMatrixView_" + ellpack ).c_str() );
   export_SparseMatrixView< typename Sparse_SE< T >::ViewType, SparseBase_SE< T > >(
      m, ( "SparseMatrixView_" + slicedellpack ).c_str() );
   export_SparseMatrixView< typename Sparse_CSR< T >::ConstViewType, SparseBase_CSR_const< T > >(
      m, ( "SparseMatrixView_" + csr + "_const" ).c_str() );
   export_SparseMatrixView< typename Sparse_Ell< T >::ConstViewType, SparseBase_Ell_const< T > >(
      m, ( "SparseMatrixView_" + ellpack + "_const" ).c_str() );
   export_SparseMatrixView< typename Sparse_SE< T >::ConstViewType, SparseBase_SE_const< T > >(
      m, ( "SparseMatrixView_" + slicedellpack + "_const" ).c_str() );

   // Same-device copy functions
   def_copySparseMatrix< Sparse_CSR< T >, Sparse_Ell< T > >( m );
   def_copySparseMatrix< Sparse_Ell< T >, Sparse_CSR< T > >( m );
   def_copySparseMatrix< Sparse_CSR< T >, Sparse_SE< T > >( m );
   def_copySparseMatrix< Sparse_SE< T >, Sparse_CSR< T > >( m );
   def_copySparseMatrix< Sparse_Ell< T >, Sparse_SE< T > >( m );
   def_copySparseMatrix< Sparse_SE< T >, Sparse_Ell< T > >( m );

   // Dense matrices
   export_DenseMatrix< Dense_RowMajor< T >, DenseBase_RowMajor< T > >( m, ( "DenseMatrix_" + rowmajor ).c_str() );
   export_DenseMatrix< Dense_ColumnMajor< T >, DenseBase_ColumnMajor< T > >( m, ( "DenseMatrix_" + columnmajor ).c_str() );

   export_DenseRowView< typename Dense_RowMajor< T >::RowView >( m, ( "DenseMatrixRowView_" + rowmajor ).c_str() );
   export_DenseRowView< typename Dense_RowMajor< T >::ConstRowView >( m, ( "DenseMatrixConstRowView_" + rowmajor ).c_str() );
   export_DenseRowView< typename Dense_ColumnMajor< T >::RowView >( m, ( "DenseMatrixRowView_" + columnmajor ).c_str() );
   export_DenseRowView< typename Dense_ColumnMajor< T >::ConstRowView >(
      m, ( "DenseMatrixConstRowView_" + columnmajor ).c_str() );

   export_DenseMatrixView< typename Dense_RowMajor< T >::ViewType, DenseBase_RowMajor< T > >(
      m, ( "DenseMatrixView_" + rowmajor ).c_str() );
   export_DenseMatrixView< typename Dense_RowMajor< T >::ConstViewType, DenseBase_RowMajor< std::add_const_t< T > > >(
      m, ( "DenseMatrixView_" + rowmajor + "_const" ).c_str() );
   export_DenseMatrixView< typename Dense_ColumnMajor< T >::ViewType, DenseBase_ColumnMajor< T > >(
      m, ( "DenseMatrixView_" + columnmajor ).c_str() );
   export_DenseMatrixView< typename Dense_ColumnMajor< T >::ConstViewType, DenseBase_ColumnMajor< std::add_const_t< T > > >(
      m, ( "DenseMatrixView_" + columnmajor + "_const" ).c_str() );
}

// Python module definition
NB_MODULE( _matrices, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers" );

   export_format_tags( m );
   export_organizations( m );
   export_segments_types( m );

   export_value_type< RealType >( m, "float" );
   export_value_type< ComplexType >( m, "complex" );
}

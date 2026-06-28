#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include <TNL/Algorithms/Segments/CSR.h>
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

using CSR_host = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, CSR >;
using E_host = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, Ellpack >;
using SE_host =
   TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, SlicedEllpack >;

using Dense_host = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Host, IndexType >;

// Base class types (mutable) — SparseMatrix::Base is private in TNL, so we
// construct the type from public typedefs.
using SparseMatrixBase_CSR_host = TNL::Matrices::SparseMatrixBase<
   RealType,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename CSR_host::SegmentsType::ViewType,
   RealType >;
using SparseMatrixBase_E_host = TNL::Matrices::SparseMatrixBase<
   RealType,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename E_host::SegmentsType::ViewType,
   RealType >;
using SparseMatrixBase_SE_host = TNL::Matrices::SparseMatrixBase<
   RealType,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename SE_host::SegmentsType::ViewType,
   RealType >;
using DenseMatrixBase_host =
   TNL::Matrices::DenseMatrixBase< RealType, TNL::Devices::Host, IndexType, Dense_host::getOrganization() >;

// Base class types (const — for const views, Real is const-qualified and
// SegmentsView uses ConstViewType per SparseMatrixView's inheritance)
using SparseMatrixBase_CSR_host_const = TNL::Matrices::SparseMatrixBase<
   const RealType,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename CSR_host::SegmentsType::ViewType::ConstViewType,
   RealType >;
using SparseMatrixBase_E_host_const = TNL::Matrices::SparseMatrixBase<
   const RealType,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename E_host::SegmentsType::ViewType::ConstViewType,
   RealType >;
using SparseMatrixBase_SE_host_const = TNL::Matrices::SparseMatrixBase<
   const RealType,
   TNL::Devices::Host,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename SE_host::SegmentsType::ViewType::ConstViewType,
   RealType >;
using DenseMatrixBase_host_const =
   TNL::Matrices::DenseMatrixBase< const RealType, TNL::Devices::Host, IndexType, Dense_host::getOrganization() >;

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
export_base_classes( nb::module_& m )
{
   export_SparseMatrixBaseClass< SparseMatrixBase_CSR_host >( m, "SparseMatrixBase_float_CSR" );
   export_SparseMatrixBaseClass< SparseMatrixBase_E_host >( m, "SparseMatrixBase_float_Ellpack" );
   export_SparseMatrixBaseClass< SparseMatrixBase_SE_host >( m, "SparseMatrixBase_float_SlicedEllpack" );

   export_SparseMatrixBaseClass< SparseMatrixBase_CSR_host_const >( m, "SparseMatrixBase_float_CSR_const" );
   export_SparseMatrixBaseClass< SparseMatrixBase_E_host_const >( m, "SparseMatrixBase_float_Ellpack_const" );
   export_SparseMatrixBaseClass< SparseMatrixBase_SE_host_const >( m, "SparseMatrixBase_float_SlicedEllpack_const" );

   export_DenseMatrixBaseClass< DenseMatrixBase_host >( m, "DenseMatrixBase_float" );
   export_DenseMatrixBaseClass< DenseMatrixBase_host_const >( m, "DenseMatrixBase_float_const" );
}

void
export_SparseMatrices( nb::module_& m )
{
   export_Matrix< CSR_host, SparseMatrixBase_CSR_host >( m, "SparseMatrix_float_CSR" );
   export_Matrix< E_host, SparseMatrixBase_E_host >( m, "SparseMatrix_float_Ellpack" );
   export_Matrix< SE_host, SparseMatrixBase_SE_host >( m, "SparseMatrix_float_SlicedEllpack" );

   // NOTE: all exported formats (CSR, Ellpack, SlicedEllpack) use the same
   // SegmentView, so the RowView and ConstRowView are also the same types in all
   // three formats
   export_RowView< typename CSR_host::RowView >( m, "SparseMatrixRowView" );
   export_RowView< typename CSR_host::ConstRowView >( m, "SparseMatrixConstRowView" );

   export_SparseMatrixView< typename CSR_host::ViewType, SparseMatrixBase_CSR_host >( m, "SparseMatrixView_float_CSR" );
   export_SparseMatrixView< typename E_host::ViewType, SparseMatrixBase_E_host >( m, "SparseMatrixView_float_Ellpack" );
   export_SparseMatrixView< typename SE_host::ViewType, SparseMatrixBase_SE_host >( m, "SparseMatrixView_float_SlicedEllpack" );
   export_SparseMatrixView< typename CSR_host::ConstViewType, SparseMatrixBase_CSR_host_const >(
      m, "SparseMatrixView_float_CSR_const" );
   export_SparseMatrixView< typename E_host::ConstViewType, SparseMatrixBase_E_host_const >(
      m, "SparseMatrixView_float_Ellpack_const" );
   export_SparseMatrixView< typename SE_host::ConstViewType, SparseMatrixBase_SE_host_const >(
      m, "SparseMatrixView_float_SlicedEllpack_const" );

   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, E_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, CSR_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, SE_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, CSR_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, SE_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, E_host > );
}

void
export_DenseMatrices( nb::module_& m )
{
   export_DenseMatrix< Dense_host, DenseMatrixBase_host >( m, "DenseMatrix_float" );

   export_DenseRowView< typename Dense_host::RowView >( m, "DenseMatrixRowView" );
   export_DenseRowView< typename Dense_host::ConstRowView >( m, "DenseMatrixConstRowView" );

   export_DenseMatrixView< typename Dense_host::ViewType, DenseMatrixBase_host >( m, "DenseMatrixView_float" );
   export_DenseMatrixView< typename Dense_host::ConstViewType, DenseMatrixBase_host_const >( m, "DenseMatrixView_float_const" );
}

// Python module definition
NB_MODULE( _matrices, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers" );

   export_format_tags( m );
   export_base_classes( m );
   export_SparseMatrices( m );
   export_DenseMatrices( m );
}

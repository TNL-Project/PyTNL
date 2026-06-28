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

using CSR_cuda = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, CSR >;
using E_cuda = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, Ellpack >;
using SE_cuda =
   TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, SlicedEllpack >;

using Dense_cuda = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Cuda, IndexType >;

// Base class types (mutable) — SparseMatrix::Base is private in TNL, so we
// construct the type from public typedefs.
using SparseMatrixBase_CSR_cuda = TNL::Matrices::SparseMatrixBase<
   RealType,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename CSR_cuda::SegmentsType::ViewType,
   RealType >;
using SparseMatrixBase_E_cuda = TNL::Matrices::SparseMatrixBase<
   RealType,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename E_cuda::SegmentsType::ViewType,
   RealType >;
using SparseMatrixBase_SE_cuda = TNL::Matrices::SparseMatrixBase<
   RealType,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename SE_cuda::SegmentsType::ViewType,
   RealType >;
using DenseMatrixBase_cuda =
   TNL::Matrices::DenseMatrixBase< RealType, TNL::Devices::Cuda, IndexType, Dense_cuda::getOrganization() >;

// Base class types (const — for const views, Real is const-qualified and
// SegmentsView uses ConstViewType per SparseMatrixView's inheritance)
using SparseMatrixBase_CSR_cuda_const = TNL::Matrices::SparseMatrixBase<
   const RealType,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename CSR_cuda::SegmentsType::ViewType::ConstViewType,
   RealType >;
using SparseMatrixBase_E_cuda_const = TNL::Matrices::SparseMatrixBase<
   const RealType,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename E_cuda::SegmentsType::ViewType::ConstViewType,
   RealType >;
using SparseMatrixBase_SE_cuda_const = TNL::Matrices::SparseMatrixBase<
   const RealType,
   TNL::Devices::Cuda,
   IndexType,
   TNL::Matrices::GeneralMatrix,
   typename SE_cuda::SegmentsType::ViewType::ConstViewType,
   RealType >;
using DenseMatrixBase_cuda_const =
   TNL::Matrices::DenseMatrixBase< const RealType, TNL::Devices::Cuda, IndexType, Dense_cuda::getOrganization() >;

void
export_base_classes( nb::module_& m )
{
   export_SparseMatrixBaseClass< SparseMatrixBase_CSR_cuda >( m, "SparseMatrixBase_float_CSR" );
   export_SparseMatrixBaseClass< SparseMatrixBase_E_cuda >( m, "SparseMatrixBase_float_Ellpack" );
   export_SparseMatrixBaseClass< SparseMatrixBase_SE_cuda >( m, "SparseMatrixBase_float_SlicedEllpack" );

   export_SparseMatrixBaseClass< SparseMatrixBase_CSR_cuda_const >( m, "SparseMatrixBase_float_CSR_const" );
   export_SparseMatrixBaseClass< SparseMatrixBase_E_cuda_const >( m, "SparseMatrixBase_float_Ellpack_const" );
   export_SparseMatrixBaseClass< SparseMatrixBase_SE_cuda_const >( m, "SparseMatrixBase_float_SlicedEllpack_const" );

   export_DenseMatrixBaseClass< DenseMatrixBase_cuda >( m, "DenseMatrixBase_float" );
   export_DenseMatrixBaseClass< DenseMatrixBase_cuda_const >( m, "DenseMatrixBase_float_const" );
}

void
export_SparseMatrices( nb::module_& m )
{
   export_Matrix< CSR_cuda, SparseMatrixBase_CSR_cuda >( m, "SparseMatrix_float_CSR" );
   export_Matrix< E_cuda, SparseMatrixBase_E_cuda >( m, "SparseMatrix_float_Ellpack" );
   export_Matrix< SE_cuda, SparseMatrixBase_SE_cuda >( m, "SparseMatrix_float_SlicedEllpack" );

   // NOTE: all exported formats (CSR, Ellpack, SlicedEllpack) use the same
   // SegmentView, so the RowView and ConstRowView are also the same types in all
   // three formats
   export_RowView< typename CSR_cuda::RowView >( m, "SparseMatrixRowView" );
   export_RowView< typename CSR_cuda::ConstRowView >( m, "SparseMatrixConstRowView" );

   export_SparseMatrixView< typename CSR_cuda::ViewType, SparseMatrixBase_CSR_cuda >( m, "SparseMatrixView_float_CSR" );
   export_SparseMatrixView< typename E_cuda::ViewType, SparseMatrixBase_E_cuda >( m, "SparseMatrixView_float_Ellpack" );
   export_SparseMatrixView< typename SE_cuda::ViewType, SparseMatrixBase_SE_cuda >( m, "SparseMatrixView_float_SlicedEllpack" );
   export_SparseMatrixView< typename CSR_cuda::ConstViewType, SparseMatrixBase_CSR_cuda_const >(
      m, "SparseMatrixView_float_CSR_const" );
   export_SparseMatrixView< typename E_cuda::ConstViewType, SparseMatrixBase_E_cuda_const >(
      m, "SparseMatrixView_float_Ellpack_const" );
   export_SparseMatrixView< typename SE_cuda::ConstViewType, SparseMatrixBase_SE_cuda_const >(
      m, "SparseMatrixView_float_SlicedEllpack_const" );

   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_cuda, E_cuda > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_cuda, CSR_cuda > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_cuda, SE_cuda > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_cuda, CSR_cuda > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_cuda, SE_cuda > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_cuda, E_cuda > );
}

void
export_DenseMatrices( nb::module_& m )
{
   export_DenseMatrix< Dense_cuda, DenseMatrixBase_cuda >( m, "DenseMatrix_float" );

   export_DenseRowView< typename Dense_cuda::RowView >( m, "DenseMatrixRowView" );
   export_DenseRowView< typename Dense_cuda::ConstRowView >( m, "DenseMatrixConstRowView" );

   export_DenseMatrixView< typename Dense_cuda::ViewType, DenseMatrixBase_cuda >( m, "DenseMatrixView_float" );
   export_DenseMatrixView< typename Dense_cuda::ConstViewType, DenseMatrixBase_cuda_const >( m, "DenseMatrixView_float_const" );
}

// Python module definition
NB_MODULE( _matrices_cuda, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers_cuda" );

   // format tags are not exported here — they are device-independent and
   // defined only in the Host module (_matrices), which __init__.py always
   // imports before any CUDA usage
   export_base_classes( m );
   export_SparseMatrices( m );
   export_DenseMatrices( m );
}

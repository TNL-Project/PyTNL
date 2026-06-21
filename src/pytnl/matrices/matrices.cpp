#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseOperations.h>

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
export_SparseMatrices( nb::module_& m )
{
   export_Matrix< CSR_host >( m, "SparseMatrix_float_CSR" );
   export_Matrix< E_host >( m, "SparseMatrix_float_Ellpack" );
   export_Matrix< SE_host >( m, "SparseMatrix_float_SlicedEllpack" );

   // NOTE: all exported formats (CSR, Ellpack, SlicedEllpack) use the same
   // SegmentView, so the RowView and ConstRowView are also the same types in all
   // three formats
   export_RowView< typename CSR_host::RowView >( m, "SparseMatrixRowView" );
   export_RowView< typename CSR_host::ConstRowView >( m, "SparseMatrixConstRowView" );

   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, E_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, CSR_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, SE_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, CSR_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, SE_host > );
   m.def( "copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, E_host > );
}

// Python module definition
NB_MODULE( _matrices, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers" );

   export_format_tags( m );
   export_SparseMatrices( m );
}

// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "SparseMatrix.h"

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>

using CSR_host = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< double, TNL::Devices::Host, int >;
using CSR_cuda = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< double, TNL::Devices::Cuda, int >;
using E_host = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::Ellpack< double, TNL::Devices::Host, int >;
using E_cuda = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::Ellpack< double, TNL::Devices::Cuda, int >;
using SE_host = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< double, TNL::Devices::Host, int >;
using SE_cuda = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< double, TNL::Devices::Cuda, int >;

void export_SparseMatrices( py::module & m )
{
    // TODO: This stop working after adding template parameter KernelType to Legacy::CSR
    //export_Matrix< CSR_host >( m, "CSR" );
    export_Matrix< E_host   >( m, "Ellpack" );
    export_Matrix< SE_host  >( m, "SlicedEllpack" );

    // TODO: copySparseMatrix does not work with Legacy matrices anymore
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, E_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, CSR_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, SE_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, CSR_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, SE_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, E_host >);
}

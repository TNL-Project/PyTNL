// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "SparseMatrix.h"

#include <TNL/Matrices/Legacy/CSR.h>
#include <TNL/Matrices/Legacy/Ellpack.h>
#include <TNL/Matrices/Legacy/SlicedEllpack.h>

using CSR_host = TNL::Matrices::Legacy::CSR< double, TNL::Devices::Host, int >;
using CSR_cuda = TNL::Matrices::Legacy::CSR< double, TNL::Devices::Cuda, int >;
using E_host = TNL::Matrices::Legacy::Ellpack< double, TNL::Devices::Host, int >;
using E_cuda = TNL::Matrices::Legacy::Ellpack< double, TNL::Devices::Cuda, int >;
using SE_host = TNL::Matrices::Legacy::SlicedEllpack< double, TNL::Devices::Host, int >;
using SE_cuda = TNL::Matrices::Legacy::SlicedEllpack< double, TNL::Devices::Cuda, int >;

void export_SparseMatrices( py::module & m )
{
    // TODO: This stop working after adding template parameter KernelType to Legacy::CSR
    //export_Matrix< CSR_host >( m, "CSR" );
    export_Matrix< E_host   >( m, "Ellpack" );
    export_Matrix< SE_host  >( m, "SlicedEllpack" );

    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, E_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, CSR_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, SE_host >);
    //m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, CSR_host >);
    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, SE_host >);
    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, E_host >);
}

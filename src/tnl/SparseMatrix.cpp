// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "SparseMatrix.h"

#include <TNL/Matrices/CSR.h>
#include <TNL/Matrices/Ellpack.h>
#include <TNL/Matrices/SlicedEllpack.h>

using CSR_host = TNL::Matrices::CSR< double, TNL::Devices::Host, int >;
using CSR_cuda = TNL::Matrices::CSR< double, TNL::Devices::Cuda, int >;
using E_host = TNL::Matrices::Ellpack< double, TNL::Devices::Host, int >;
using E_cuda = TNL::Matrices::Ellpack< double, TNL::Devices::Cuda, int >;
using SE_host = TNL::Matrices::SlicedEllpack< double, TNL::Devices::Host, int >;
using SE_cuda = TNL::Matrices::SlicedEllpack< double, TNL::Devices::Cuda, int >;

void export_SparseMatrices( py::module & m )
{
    export_Matrix< CSR_host >( m, "CSR" );
    export_Matrix< E_host   >( m, "Ellpack" );
    export_Matrix< SE_host  >( m, "SlicedEllpack" );

    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, E_host >);
    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, CSR_host >);
    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, SE_host >);
    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, CSR_host >);
    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, SE_host >);
    m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, E_host >);
}

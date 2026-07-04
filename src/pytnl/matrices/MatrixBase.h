#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Matrices/MatrixBase.h>

// Registering MatrixBase as a nanobind base class would not be useful/effective:
// the Organization template parameter differs across formats (CSR=RowMajor,
// Ellpack/Dense on CUDA=ColumnMajor), producing different C++ types that cannot
// share a single nanobind base class.
template< typename Matrix, typename MatrixClass >
void
def_MatrixBaseMethods( MatrixClass& matrix )
{
   matrix.def( "getRows", &Matrix::getRows )
      .def( "getColumns", &Matrix::getColumns )
      .def( "getAllocatedElementsCount", &Matrix::getAllocatedElementsCount )
      .def_static( "isBinary", &Matrix::isBinary )
      .def_static( "isSymmetric", &Matrix::isSymmetric )
      .def_static( "getOrganization", &Matrix::getOrganization );
}

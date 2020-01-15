#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <TNL/String.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/Legacy/CSR.h>

template< typename Matrix >
struct SpecificExports
{
    template< typename Scope >
    static void exec( Scope & s ) {}
};

template< typename Real, typename Device, typename Index >
struct SpecificExports< TNL::Matrices::CSR< Real, Device, Index > >
{
    template< typename Scope >
    static void exec( Scope & s )
    {
        using Matrix = TNL::Matrices::CSR< Real, Device, Index >;

        s.def("getRowPointers",   py::overload_cast<>(&Matrix::getRowPointers),   py::return_value_policy::reference_internal);
        s.def("getColumnIndexes", py::overload_cast<>(&Matrix::getColumnIndexes), py::return_value_policy::reference_internal);
        s.def("getValues",        py::overload_cast<>(&Matrix::getValues),        py::return_value_policy::reference_internal);
    }
};

template< typename MatrixRow >
void export_MatrixRow( py::module & m, const char* name )
{
    // guard against duplicate to-Python converters for the same type
    static bool defined = false;
    if( ! defined ) {
        py::class_< MatrixRow >( m, name )
            .def("setElement", &MatrixRow::setElement)
            .def("getElementColumn", &MatrixRow::getElementColumn, py::return_value_policy::reference_internal)
            .def("getElementValue", &MatrixRow::getElementValue, py::return_value_policy::reference_internal)
//            .def(py::self_ns::str(py::self_ns::self))
        ;
        defined = true;
    }
}

template< typename Matrix >
void export_Matrix( py::module & m, const char* name )
{
    typename Matrix::MatrixRow (Matrix::* _getRow)(typename Matrix::IndexType) = &Matrix::getRow;

    using VectorType = TNL::Containers::Vector< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;

    void (Matrix::* _getCompressedRowLengths)(typename Matrix::CompressedRowLengthsVectorView) const = &Matrix::getCompressedRowLengths;

    auto matrix = py::class_< Matrix, TNL::Object >( m, name )
        .def(py::init<>())
        // overloads (defined in Object)
        .def_static("getSerializationType", &Matrix::getSerializationType)
        .def("getSerializationTypeVirtual", &Matrix::getSerializationTypeVirtual)
        .def("print", &Matrix::print)
        .def("__str__", []( Matrix & m ) {
                std::stringstream ss;
                ss << m;
                return ss.str();
            } )

        // Matrix
        .def("setDimensions",           &Matrix::setDimensions)
        .def("setCompressedRowLengths", &Matrix::setCompressedRowLengths)
        .def("getRowLength",            &Matrix::getRowLength)
        .def("getCompressedRowLengths", _getCompressedRowLengths)
        // TODO: export for more types
        .def("setLike",                 &Matrix::template setLike< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >)
        .def("getAllocatedElementsCount", &Matrix::getAllocatedElementsCount)
        .def("getNumberOfNonzeroMatrixElements", &Matrix::getNumberOfNonzeroMatrixElements)
        .def("reset",                   &Matrix::reset)
        .def("getRows",                 &Matrix::getRows)
        .def("getColumns",              &Matrix::getColumns)
        .def("setElement",              &Matrix::setElement)
        .def("addElement",              &Matrix::addElement)
        // setRow and addRow operate on pointers
        //.def("setRow",                  &Matrix::setRow)
        //.def("addRow",                  &Matrix::addRow)
        .def("getElement",              &Matrix::getElement)
        // TODO: operator== and operator!= are general and very slow
        
        // Sparse
        .def("getMaxRowLength",     &Matrix::getMaxRowLength)
        .def("getPaddingIndex",     &Matrix::getPaddingIndex)
        // TODO: this one is empty in the C++ code
//        .def("printStructure",      &Matrix::printStructure)

        // specific to each format, but with common interface
        .def("getRow",              _getRow)
        // TODO: export for more types
        .def("rowVectorProduct",    &Matrix::template rowVectorProduct< VectorType >)
        .def("vectorProduct",       &Matrix::template vectorProduct< VectorType, VectorType >)
        // TODO: these two don't work
        //.def("addMatrix",           &Matrix::addMatrix)
        //.def("getTransposition",    &Matrix::getTransposition)
        .def("performSORIteration", &Matrix::template performSORIteration< VectorType, VectorType >)
//        .def("assign",              &Matrix::operator=)
        .def("assign", []( Matrix& matrix, const Matrix& other ) -> Matrix& {
                return matrix = other;
            })
    ;

    // export format-specific methods
    SpecificExports< Matrix >::exec( matrix );

    export_MatrixRow< typename Matrix::MatrixRow >( m, "MatrixRow" );
}

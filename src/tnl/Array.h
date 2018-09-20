#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include "../tnl_indexing.h"

#include <TNL/Containers/Array.h>

template< typename ArrayType >
void export_Array(py::module & m, const char* name)
{
    auto array = py::class_<ArrayType, TNL::Object>(m, name, py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<int>())
        .def_static("getType",              &ArrayType::getType)
        .def("getTypeVirtual",              &ArrayType::getTypeVirtual)
        .def_static("getSerializationType", &ArrayType::getSerializationType)
        .def("getSerializationTypeVirtual", &ArrayType::getSerializationTypeVirtual)
        .def("setSize", &ArrayType::setSize)
        .def("setLike", &ArrayType::template setLike<ArrayType>)
        .def("swap", &ArrayType::swap)
        .def("reset", &ArrayType::reset)
        .def("getSize", &ArrayType::getSize)
        .def("setElement", &ArrayType::setElement)
        .def("getElement", &ArrayType::getElement)
        // operator=
        .def("assign", []( ArrayType& array, const ArrayType& other ) -> ArrayType& {
                return array = other;
            })
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("setValue", &ArrayType::setValue)

        .def("__str__", []( ArrayType & a ) {
                std::stringstream ss;
                ss << a;
                return ss.str();
            } )

        // Python buffer protocol: http://pybind11.readthedocs.io/en/master/advanced/pycpp/numpy.html
        .def_buffer( [](ArrayType & a) -> py::buffer_info {
            return py::buffer_info(
                // Pointer to buffer
                a.getData(),
                // Size of one scalar
                sizeof( typename ArrayType::ValueType ),
                // Python struct-style format descriptor
                py::format_descriptor< typename ArrayType::ValueType >::format(),
                // Number of dimensions
                1,
                // Buffer dimensions
                { a.getSize() },
                // Strides (in bytes) for each index
                { sizeof( typename ArrayType::ValueType ) }
            );
        })
    ;

    tnl_indexing< ArrayType >( array );
    tnl_slice_indexing< ArrayType >( array );
}

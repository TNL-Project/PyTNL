#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include "../tnl_indexing.h"

// needed for discovery of operator<< for tnlStaticArray
#include <TNL/Containers/StaticArray.h>

template< typename VectorType, typename Scope >
void export_StaticVector( Scope & scope, const char* name )
{
    using RealType = typename VectorType::RealType;

    auto vector = py::class_<VectorType>(scope, name)
        .def(py::init< typename VectorType::RealType >())
        .def(py::init< VectorType >())
        .def_static("getType", &VectorType::getType)
        .def("getSize", &VectorType::getSize)
        .def("assign", &VectorType::operator=)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("setValue", &VectorType::setValue)
        // TODO: pybind11
        // explicit namespace resolution is necessary, see http://stackoverflow.com/a/3084341/4180822
//        .def(py::self_ns::str(py::self))
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= typename VectorType::RealType())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * typename VectorType::RealType())
        .def(py::self * py::self)
        .def(py::self < py::self)
        .def(py::self > py::self)
        .def(py::self <= py::self)
        .def(py::self >= py::self)
    ;

    tnl_indexing< VectorType >( vector );
}

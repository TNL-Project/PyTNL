#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include <TNL/Containers/Vector.h>

template< typename ArrayType, typename VectorType >
void export_Vector(py::module & m, const char* name)
{
    // function pointers for overloaded methods
    void (VectorType::* _addElement1)(const typename VectorType::IndexType,
                                      const typename VectorType::RealType &)
        = &VectorType::addElement;
    void (VectorType::* _addElement2)(const typename VectorType::IndexType,
                                      const typename VectorType::RealType &,
                                      const typename VectorType::RealType &)
        = &VectorType::addElement;

    py::class_<VectorType, ArrayType>(m, name)
        .def(py::init<>())
        .def(py::init<int>())
        .def_static("getType",              &VectorType::getType)
        .def("getTypeVirtual",              &VectorType::getTypeVirtual)
        .def_static("getSerializationType", &VectorType::getSerializationType)
        .def("getSerializationTypeVirtual", &VectorType::getSerializationTypeVirtual)
        .def("addElement", _addElement1)
        .def("addElement", _addElement2)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= typename VectorType::RealType())
        .def(py::self /= typename VectorType::RealType())
        .def("max", &VectorType::max)
        .def("min", &VectorType::min)
        .def("absMax", &VectorType::absMax)
        .def("absMin", &VectorType::absMin)
        .def("lpNorm", &VectorType::template lpNorm<double, double>)
        .def("sum", &VectorType::template sum<double>)
        .def("differenceMax", &VectorType::template differenceMax<VectorType>)
        .def("differenceMin", &VectorType::template differenceMin<VectorType>)
        .def("differenceAbsMax", &VectorType::template differenceAbsMax<VectorType>)
        .def("differenceAbsMin", &VectorType::template differenceAbsMin<VectorType>)
        .def("differenceLpNorm", &VectorType::template differenceLpNorm<double, VectorType, double>)
        .def("differenceSum", &VectorType::template differenceSum<double, VectorType>)
        .def("scalarProduct", &VectorType::template scalarProduct<VectorType>)
        .def("addVector", &VectorType::template addVector<VectorType>)
        .def("addVectors", &VectorType::template addVectors<VectorType>)
    ;
}

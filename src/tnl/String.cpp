#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include <TNL/String.h>
#include <TNL/File.h>

void export_String( py::module & m )
{
    // function pointers for overloaded methods
//    const char* (TNL::String::* _String_getString)() const = &TNL::String::getString;
    bool (TNL::String::* _String_save)(TNL::File &) const = &TNL::String::save;
    bool (TNL::String::* _String_load)(TNL::File &)       = &TNL::String::load;

    py::class_<TNL::String>(m, "String")
        .def(py::init<const char*>())
        .def(py::init<const char*, int>())
        .def(py::init<const char*, int, int>())
        .def(py::init<int>())
        .def(py::init<double>())
        .def_static("getType", &TNL::String::getType)
        // __str__ (uses operator<<)
        // explicit namespace resolution is necessary, see http://stackoverflow.com/a/3084341/4180822
//        .def(py::self_ns::str(py::self_ns::self))
        // TODO: operator[]
//        .def(vector_indexing_suite<TNL::String>())
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def("getLength", &TNL::String::getLength)
        .def("__len__", &TNL::String::getLength)
        // FIXME
//        .def("replace", &TNL::String::replace)
        .def("save", _String_save)
        .def("load", _String_load)
    ;
}

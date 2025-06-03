#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <TNL/Containers/Vector.h>

template< typename ArrayType, typename VectorType >
void
export_Vector( py::module& m, const char* name )
{
   using RealType = typename VectorType::RealType;

   py::class_< VectorType, ArrayType >( m, name )
      .def( py::init<>() )
      .def( py::init< int >() )
      .def_static( "getSerializationType", &VectorType::getSerializationType )
      .def( "getSerializationTypeVirtual", &VectorType::getSerializationTypeVirtual )
      .def( py::self == py::self )
      .def( py::self != py::self )
      .def( py::self += py::self )
      .def( py::self -= py::self )
      .def( py::self *= py::self )
      .def( py::self /= py::self )
      .def( py::self += RealType() )
      .def( py::self -= RealType() )
      .def( py::self *= RealType() )
      .def( py::self /= RealType() )
      .def( py::self + py::self )
      .def( py::self - py::self )
      .def( py::self * py::self )
      .def( py::self / py::self )
      .def( py::self + RealType() )
      .def( py::self - RealType() )
      .def( py::self * RealType() )
      .def( py::self / RealType() )
      // Deepcopy support https://pybind11.readthedocs.io/en/stable/advanced/classes.html#deepcopy-support
      .def( "__copy__",
            []( const VectorType& self )
            {
               return VectorType( self );
            } )
      .def(
         "__deepcopy__",
         []( const VectorType& self, py::dict )
         {
            return VectorType( self );
         },
         py::arg( "memo" ) );
}

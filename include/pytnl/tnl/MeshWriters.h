#pragma once

#include <pytnl/nanobind.h>
#include <pytnl/iostream_caster.h>

#include <TNL/Meshes/VTKTraits.h>

// helper struct is needed to ensure correct initialization order in the
// PyWriter constructor
struct PyOstreamHelper
{
   nb::object obj;
   pystreambuf::ostream str;

   PyOstreamHelper( nb::object src ) : obj( nb::borrow( src ) ), str( obj ) {}
};

template< typename Writer, TNL::Meshes::VTK::FileFormat default_format >
struct PyWriter : public PyOstreamHelper, public Writer
{
   PyWriter( nb::object src, TNL::Meshes::VTK::FileFormat format = default_format ) : PyOstreamHelper( src ), Writer( str ) {}
};

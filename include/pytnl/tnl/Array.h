#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

// including pybind11/numpy.h is needed for the specializations of
// py::format_descriptor for enum types, see
// https://github.com/pybind/pybind11/issues/2135
#include <pybind11/numpy.h>

#include <pytnl/tnl_indexing.h>

#include <TNL/Containers/Array.h>

template< typename ArrayType >
void
export_Array( py::module& m, const char* name )
{
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   auto array =  //
      py::class_< ArrayType >( m, name, py::buffer_protocol() )
         // Constructors
         .def( py::init<>() )
         .def( py::init< IndexType >(), py::arg( "size" ) )
         .def( py::init< IndexType, ValueType >(), py::arg( "size" ), py::arg( "value" ) )

         // Typedefs
         .def_property_readonly_static(  //
            "IndexType",
            []( py::object )
            {
               // py::type::of<> does not handle generic types like int, float, etc.
               // https://github.com/pybind/pybind11/issues/2486
               if constexpr( std::is_integral_v< IndexType > ) {
                  return py::type::of( py::int_() );
               }
               else {
                  return py::type::of< IndexType >();
               }
            } )
         .def_property_readonly_static(  //
            "ValueType",
            []( py::object )
            {
               // py::type::of<> does not handle generic types like int, float, etc.
               // https://github.com/pybind/pybind11/issues/2486
               if constexpr( std::is_same_v< ValueType, bool > ) {
                  return py::type::of( py::bool_() );
               }
               else if constexpr( std::is_integral_v< ValueType > ) {
                  return py::type::of( py::int_() );
               }
               else if constexpr( std::is_floating_point_v< ValueType > ) {
                  return py::type::of( py::float_() );
               }
               else {
                  return py::type::of< ValueType >();
               }
            } )

         // Size management
         .def( "getSize", &ArrayType::getSize )
         .def( "setSize", &ArrayType::setSize, py::arg( "size" ) )
         .def( "setLike", &ArrayType::template setLike< ArrayType > )
         .def( "resize", py::overload_cast< IndexType >( &ArrayType::resize ), py::arg( "size" ) )
         .def(
            "resize", py::overload_cast< IndexType, ValueType >( &ArrayType::resize ), py::arg( "size" ), py::arg( "value" ) )
         .def( "swap", &ArrayType::swap )
         .def( "reset", &ArrayType::reset )
         .def( "empty", &ArrayType::empty )

         // Data access
         .def(
            "setElement",
            []( ArrayType& array, typename ArrayType::IndexType i, typename ArrayType::ValueType value )
            {
               if( i < 0 || i >= array.getSize() )
                  throw py::index_error( "index " + std::to_string( i ) + " is out-of-bounds for given array with size "
                                         + std::to_string( array.getSize() ) );
               array.setElement( i, value );
            },
            py::arg( "i" ),
            py::arg( "value" ) )
         .def(
            "getElement",
            []( const ArrayType& array, typename ArrayType::IndexType i )
            {
               if( i < 0 || i >= array.getSize() )
                  throw py::index_error( "index " + std::to_string( i ) + " is out-of-bounds for given array with size "
                                         + std::to_string( array.getSize() ) );
               return array.getElement( i );
            },
            py::arg( "i" ) )

         // Assignment
         .def( "assign",
               []( ArrayType& array, const ArrayType& other ) -> ArrayType&
               {
                  return array = other;
               } )

         // Comparison
         .def( py::self == py::self )
         .def( py::self != py::self )

         // Fill
         .def( "setValue", &ArrayType::setValue, py::arg( "value" ), py::arg( "begin" ) = 0, py::arg( "end" ) = 0 )

         // File I/O
         .def_static( "getSerializationType", &ArrayType::getSerializationType )
         .def( "getSerializationTypeVirtual", &ArrayType::getSerializationTypeVirtual )
         .def( "save", &ArrayType::save )
         .def( "load", &ArrayType::load )

         // String representation
         .def( "__str__",
               []( ArrayType& a )
               {
                  std::stringstream ss;
                  ss << a;
                  return ss.str();
               } )

         // Deepcopy support https://pybind11.readthedocs.io/en/stable/advanced/classes.html#deepcopy-support
         .def( "__copy__",
               []( const ArrayType& self )
               {
                  return ArrayType( self );
               } )
         .def(
            "__deepcopy__",
            []( const ArrayType& self, py::dict )
            {
               return ArrayType( self );
            },
            py::arg( "memo" ) )

         // Buffer protocol for NumPy interoperability
         // http://pybind11.readthedocs.io/en/master/advanced/pycpp/numpy.html
         .def_buffer(
            []( ArrayType& a ) -> py::buffer_info
            {
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
                  { sizeof( typename ArrayType::ValueType ) } );
            } );

   tnl_indexing< ArrayType >( array );
   tnl_slice_indexing< ArrayType >( array );
}

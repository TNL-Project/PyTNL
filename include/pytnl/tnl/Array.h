#pragma once

#include <pytnl/nanobind.h>
#include <pytnl/tnl_indexing.h>

#include <TNL/Containers/Array.h>

template< typename ArrayType >
void
export_Array( nb::module_& m, const char* name )
{
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   auto array =  //
      nb::class_< ArrayType >( m, name )
         // Constructors
         .def( nb::init<>() )
         // NOTE: the nb::init<...> does not work due to list-initialization and
         //       std::list_initializer constructor in ArrayType
         .def( my_init< IndexType >(), nb::arg( "size" ) )
         .def( my_init< IndexType, ValueType >(), nb::arg( "size" ), nb::arg( "value" ) )

         // Typedefs
         .def_prop_ro_static(  //
            "IndexType",
            []( nb::object ) -> nb::object
            {
               // nb::type<> does not handle generic types like int, float, etc.
               // https://github.com/wjakob/nanobind/discussions/1070
               if constexpr( std::is_integral_v< IndexType > ) {
                  return nb::borrow( &PyLong_Type );
               }
               else {
                  return nb::type< IndexType >();
               }
            } )
         .def_prop_ro_static(  //
            "ValueType",
            []( nb::object ) -> nb::object
            {
               // nb::type<> does not handle generic types like int, float, etc.
               // https://github.com/wjakob/nanobind/discussions/1070
               if constexpr( std::is_same_v< ValueType, bool > ) {
                  return nb::borrow( &PyBool_Type );
               }
               else if constexpr( std::is_integral_v< ValueType > ) {
                  return nb::borrow( &PyLong_Type );
               }
               else if constexpr( std::is_floating_point_v< ValueType > ) {
                  return nb::borrow( &PyFloat_Type );
               }
               else {
                  return nb::type< ValueType >();
               }
            } )

         // Size management
         .def( "getSize", &ArrayType::getSize )
         .def( "setSize", &ArrayType::setSize, nb::arg( "size" ) )
         .def( "setLike", &ArrayType::template setLike< ArrayType > )
         .def( "resize", nb::overload_cast< IndexType >( &ArrayType::resize ), nb::arg( "size" ) )
         .def(
            "resize", nb::overload_cast< IndexType, ValueType >( &ArrayType::resize ), nb::arg( "size" ), nb::arg( "value" ) )
         .def( "swap", &ArrayType::swap )
         .def( "reset", &ArrayType::reset )
         .def( "empty", &ArrayType::empty )

         // Data access
         .def(
            "setElement",
            []( ArrayType& array, typename ArrayType::IndexType i, typename ArrayType::ValueType value )
            {
               if( i < 0 || i >= array.getSize() )
                  throw nb::index_error( ( "index " + std::to_string( i ) + " is out-of-bounds for given array with size "
                                           + std::to_string( array.getSize() ) )
                                            .c_str() );
               array.setElement( i, value );
            },
            nb::arg( "i" ),
            nb::arg( "value" ) )
         .def(
            "getElement",
            []( const ArrayType& array, typename ArrayType::IndexType i )
            {
               if( i < 0 || i >= array.getSize() )
                  throw nb::index_error( ( "index " + std::to_string( i ) + " is out-of-bounds for given array with size "
                                           + std::to_string( array.getSize() ) )
                                            .c_str() );
               return array.getElement( i );
            },
            nb::arg( "i" ) )

         // Assignment
         .def( "assign",
               []( ArrayType& array, const ArrayType& other ) -> ArrayType&
               {
                  return array = other;
               } )

         // Comparison
         .def( nb::self == nb::self )
         .def( nb::self != nb::self )

         // Fill
         .def( "setValue", &ArrayType::setValue, nb::arg( "value" ), nb::arg( "begin" ) = 0, nb::arg( "end" ) = 0 )

         // File I/O
         .def_static( "getSerializationType", &ArrayType::getSerializationType )
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
            []( const ArrayType& self, nb::dict )
            {
               return ArrayType( self );
            },
            nb::arg( "memo" ) )

         .def( "as_ndarray",
               []( ArrayType& self )
               {
                  return nb::ndarray< ValueType >( self.getData(), { static_cast< std::size_t >( self.getSize() ) } );
               } );

   // Buffer protocol for NumPy interoperability
   // http://pybind11.readthedocs.io/en/master/advanced/pycpp/numpy.html
   //.def_buffer(
   //   []( ArrayType& a ) -> nb::buffer_info
   //   {
   //      return nb::buffer_info(
   //         // Pointer to buffer
   //         a.getData(),
   //         // Size of one scalar
   //         sizeof( typename ArrayType::ValueType ),
   //         // Python struct-style format descriptor
   //         nb::format_descriptor< typename ArrayType::ValueType >::format(),
   //         // Number of dimensions
   //         1,
   //         // Buffer dimensions
   //         { a.getSize() },
   //         // Strides (in bytes) for each index
   //         { sizeof( typename ArrayType::ValueType ) } );
   //   } );

   tnl_indexing< ArrayType >( array );
   tnl_slice_indexing< ArrayType >( array );
}

#pragma once

#include <pytnl/pytnl.h>

#include <pytnl/containers/DistributedArray.h>

#include <TNL/Containers/DistributedVector.h>
#include <TNL/TypeTraits.h>

#include <sstream>

template< typename ArrayType, typename VectorType >
void
export_DistributedVector( nb::module_& m, const char* name )
{
   using RealType = typename VectorType::RealType;
   using DeviceType = typename VectorType::DeviceType;

   auto vector =  //
      nb::class_< VectorType, ArrayType >( m, name )
         // Typedefs
         .def_prop_ro_static(  //
            "RealType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
            {
               // nb::type<> does not handle generic types like int, float, etc.
               // https://github.com/wjakob/nanobind/discussions/1070
               if constexpr( std::is_same_v< RealType, bool > ) {
                  return nb::borrow( &PyBool_Type );
               }
               else if constexpr( std::is_integral_v< RealType > ) {
                  return nb::borrow( &PyLong_Type );
               }
               else if constexpr( std::is_floating_point_v< RealType > ) {
                  return nb::borrow( &PyFloat_Type );
               }
               else if constexpr( TNL::is_complex_v< RealType > ) {
                  return nb::borrow( &PyComplex_Type );
               }
               else {
                  return nb::type< RealType >();
               }
            } )
         .def_prop_ro_static(  //
            "LocalViewType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
            {
               return nb::type< typename VectorType::LocalViewType >();
            } )
         .def_prop_ro_static(  //
            "ConstLocalViewType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
            {
               return nb::type< typename VectorType::ConstLocalViewType >();
            } )

         // Constructors
         .def( nb::init<>() )
         .def( nb::init< const VectorType& >(), nb::arg( "other" ) )

         // Local views (overridden to return VectorView instead of ArrayView)
         .def( "getLocalView",
               []( VectorType& self )
               {
                  return self.getLocalView();
               },
               "Returns a modifiable view of the local part of the vector" )
         .def( "getConstLocalView",
               []( const VectorType& self )
               {
                  return self.getConstLocalView();
               },
               "Returns a non-modifiable view of the local part of the vector" )
         .def( "getLocalViewWithGhosts",
               []( VectorType& self )
               {
                  return self.getLocalViewWithGhosts();
               },
               "Returns a modifiable view of the local part of the vector, including ghost values" )
         .def( "getConstLocalViewWithGhosts",
               []( const VectorType& self )
               {
                  return self.getConstLocalViewWithGhosts();
               },
               "Returns a non-modifiable view of the local part of the vector, including ghost values" )

         // String representation
         .def(
            "__str__",
            []( const VectorType& self )
            {
               const auto range = self.getLocalRange();
               std::ostringstream oss;
               oss << "DistributedVector[" << distributed_value_type_name< RealType >() << ", "
                   << ( std::is_same_v< DeviceType, TNL::Devices::Cuda > ? "Cuda" : "Host" ) << "](" << self.getSize()
                   << ", localRange=(" << range.getBegin() << ", " << range.getEnd() << "), ghosts=" << self.getGhosts() << ")";
               return oss.str();
            },
            "Returns a readable string representation of the vector" )

         // Deepcopy support https://pybind11.readthedocs.io/en/stable/advanced/classes.html#deepcopy-support
         .def(
            "__copy__",
            []( const VectorType& self )
            {
               return VectorType( self );
            } )
         .def(
            "__deepcopy__",
            []( const VectorType& self, nb::typed< nb::dict, nb::str, nb::any > )
            {
               return VectorType( self );
            },
            nb::arg( "memo" ) );

   // In-place arithmetic operators. Binary operators returning a new
   // DistributedVector are not exposed because TNL does not provide
   // a constructor from the expression templates.
   if constexpr( TNL::IsScalarType< RealType >::value ) {
      vector
         .def(
            "__iadd__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self += other;
               return self;
            },
            nb::is_operator() )
         .def(
            "__iadd__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self += scalar;
               return self;
            },
            nb::is_operator() )
         .def(
            "__isub__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self -= other;
               return self;
            },
            nb::is_operator() )
         .def(
            "__isub__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self -= scalar;
               return self;
            },
            nb::is_operator() )
         .def(
            "__imul__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self *= other;
               return self;
            },
            nb::is_operator() )
         .def(
            "__imul__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self *= scalar;
               return self;
            },
            nb::is_operator() )
         .def(
            "__itruediv__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self /= other;
               return self;
            },
            nb::is_operator() )
         .def(
            "__itruediv__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self /= scalar;
               return self;
            },
            nb::is_operator() );
   }

   // The modulo operator is defined only for integral value types in C++
   if constexpr( std::is_integral_v< RealType > ) {
      vector
         .def(
            "__imod__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self %= other;
               return self;
            },
            nb::is_operator() )
         .def(
            "__imod__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self %= scalar;
               return self;
            },
            nb::is_operator() );
   }
}

#pragma once

#include <pytnl/nanobind.h>

#include <TNL/Containers/Vector.h>

template< typename ArrayType, typename VectorType >
void
export_Vector( nb::module_& m, const char* name )
{
   using IndexType = typename VectorType::IndexType;
   using RealType = typename VectorType::RealType;

   auto vector =  //
      nb::class_< VectorType, ArrayType >( m, name )
         // Constructors
         .def( nb::init<>() )
         // NOTE: the nb::init<...> does not work due to list-initialization and
         //       std::list_initializer constructor in ArrayType
         .def( my_init< IndexType >(), nb::arg( "size" ) )
         .def( my_init< IndexType, RealType >(), nb::arg( "size" ), nb::arg( "value" ) )

         // Typedefs
         .def_prop_ro_static(  //
            "RealType",
            []( nb::object ) -> nb::object
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
               else {
                  return nb::type< RealType >();
               }
            } )

         // Serialization
         .def_static( "getSerializationType", &VectorType::getSerializationType )

         // Comparison operators (Vector OP Vector)
         .def(
            "__eq__",
            []( const VectorType& self, const VectorType& other )
            {
               return self == other;
            },
            nb::is_operator() )
         .def(
            "__ne__",
            []( const VectorType& self, const VectorType& other )
            {
               return self != other;
            },
            nb::is_operator() )
         .def(
            "__lt__",
            []( const VectorType& self, const VectorType& other )
            {
               return self < other;
            },
            nb::is_operator() )
         .def(
            "__le__",
            []( const VectorType& self, const VectorType& other )
            {
               return self <= other;
            },
            nb::is_operator() )
         .def(
            "__gt__",
            []( const VectorType& self, const VectorType& other )
            {
               return self > other;
            },
            nb::is_operator() )
         .def(
            "__ge__",
            []( const VectorType& self, const VectorType& other )
            {
               return self >= other;
            },
            nb::is_operator() )

         // In-place arithmetic operators (Vector OP Vector)
         .def(
            "__iadd__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self += other;
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
            "__imul__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self *= other;
               return self;
            },
            nb::is_operator() )
         .def(
            "__idiv__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self /= other;
               return self;
            },
            nb::is_operator() )

         // In-place arithmetic operators (Vector OP Scalar)
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
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self -= scalar;
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
            "__idiv__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self /= scalar;
               return self;
            },
            nb::is_operator() )

         // Binary arithmetic operators (Vector OP Vector)
         .def(
            "__add__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self + other );
            },
            nb::is_operator() )
         .def(
            "__sub__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self - other );
            },
            nb::is_operator() )
         .def(
            "__mul__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self * other );
            },
            nb::is_operator() )
         .def(
            "__truediv__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self / other );
            },
            nb::is_operator() )

         // Binary arithmetic operators (Vector OP Scalar)
         .def(
            "__add__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self + scalar );
            },
            nb::is_operator() )
         .def(
            "__sub__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self - scalar );
            },
            nb::is_operator() )
         .def(
            "__mul__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self * scalar );
            },
            nb::is_operator() )
         .def(
            "__truediv__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self / scalar );
            },
            nb::is_operator() )

         // Reverse arithmetic operators (Scalar OP Vector)
         .def(
            "__radd__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar + self );
            },
            nb::is_operator() )
         .def(
            "__rsub__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar - self );
            },
            nb::is_operator() )
         .def(
            "__rmul__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar * self );
            },
            nb::is_operator() )
         .def(
            "__rtruediv__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar / self );
            },
            nb::is_operator() )

         // Unary Operators
         .def( "__pos__",
               []( const VectorType& self )
               {
                  return VectorType( +self );
               } )
         .def( "__neg__",
               []( const VectorType& self )
               {
                  return VectorType( -self );
               } )

         // Copy/Deepcopy
         .def( "__copy__",
               []( const VectorType& self )
               {
                  return VectorType( self );
               } )
         .def(
            "__deepcopy__",
            []( const VectorType& self, nb::dict )
            {
               return VectorType( self );
            },
            nb::arg( "memo" ) );

   // Additional operators defined only for integral value types
   if constexpr( std::is_integral_v< RealType > ) {
      vector
         // Modulo operators
         .def(
            "__mod__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self % other );
            },
            nb::is_operator() )
         .def(
            "__mod__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self % scalar );
            },
            nb::is_operator() )
         .def(
            "__rmod__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( other % self );
            },
            nb::is_operator() )
         .def(
            "__rmod__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar % self );
            },
            nb::is_operator() )
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
            nb::is_operator() )

         // Bitwise operators
         .def(
            "__and__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self & other );
            },
            nb::is_operator() )
         .def(
            "__and__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self & scalar );
            },
            nb::is_operator() )
         .def(
            "__rand__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar & self );
            },
            nb::is_operator() )

         .def(
            "__or__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self | other );
            },
            nb::is_operator() )
         .def(
            "__or__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self | scalar );
            },
            nb::is_operator() )
         .def(
            "__ror__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar | self );
            },
            nb::is_operator() )

         .def(
            "__xor__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self ^ other );
            },
            nb::is_operator() )
         .def(
            "__xor__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self ^ scalar );
            },
            nb::is_operator() )
         .def(
            "__rxor__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar ^ self );
            },
            nb::is_operator() )

         // Bitwise negation
         .def( "__invert__",
               []( const VectorType& self )
               {
                  return VectorType( ~self );
               } );
   }
}

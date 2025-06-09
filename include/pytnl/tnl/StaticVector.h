#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <pytnl/tnl_indexing.h>

template< typename VectorType, typename Scope >
void
export_StaticVector( Scope& scope, const char* name )
{
   using IndexType = typename VectorType::IndexType;
   using ValueType = typename VectorType::ValueType;
   using RealType = typename VectorType::RealType;

   auto vector =  //
      py::class_< VectorType >( scope, name )
         .def( py::init< RealType >() )
         .def( py::init< VectorType >() )
         // this enables initialization from Python lists (assuming that implicit conversion for STL containers is enabled)
         .def( py::init< std::array< ValueType, VectorType::getSize() > >() )

         // Conversion back to Python list through the STL container
         .def( "as_list",
               []( VectorType& self )
               {
                  return reinterpret_cast< std::array< ValueType, VectorType::getSize() >& >( self );
               } )

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
         .def_property_readonly_static(  //
            "RealType",
            []( py::object )
            {
               // py::type::of<> does not handle generic types like int, float, etc.
               // https://github.com/pybind/pybind11/issues/2486
               if constexpr( std::is_same_v< RealType, bool > ) {
                  return py::type::of( py::bool_() );
               }
               else if constexpr( std::is_integral_v< RealType > ) {
                  return py::type::of( py::int_() );
               }
               else if constexpr( std::is_floating_point_v< RealType > ) {
                  return py::type::of( py::float_() );
               }
               else {
                  return py::type::of< RealType >();
               }
            } )

         // Size getter
         .def_static( "getSize", &VectorType::getSize )

         // Assignment
         .def( "assign",
               []( VectorType& vector, const VectorType& other ) -> VectorType&
               {
                  return vector = other;
               } )

         // Fill
         .def( "setValue", &VectorType::setValue )

         // Comparison operators
         .def( py::self == py::self )
         .def( py::self != py::self )
         .def( py::self < py::self )
         .def( py::self > py::self )
         .def( py::self <= py::self )
         .def( py::self >= py::self )

         // TODO: pybind11
         // explicit namespace resolution is necessary, see
         // http://stackoverflow.com/a/3084341/4180822
         //        .def(py::self_ns::str(py::self))

         // Comparison operators (Vector OP Vector)
         .def(
            "__eq__",
            []( const VectorType& self, const VectorType& other )
            {
               return self == other;
            },
            py::is_operator() )
         .def(
            "__ne__",
            []( const VectorType& self, const VectorType& other )
            {
               return self != other;
            },
            py::is_operator() )
         .def(
            "__lt__",
            []( const VectorType& self, const VectorType& other )
            {
               return self < other;
            },
            py::is_operator() )
         .def(
            "__le__",
            []( const VectorType& self, const VectorType& other )
            {
               return self <= other;
            },
            py::is_operator() )
         .def(
            "__gt__",
            []( const VectorType& self, const VectorType& other )
            {
               return self > other;
            },
            py::is_operator() )
         .def(
            "__ge__",
            []( const VectorType& self, const VectorType& other )
            {
               return self >= other;
            },
            py::is_operator() )

         // In-place arithmetic operators (Vector OP Vector)
         .def(
            "__iadd__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self += other;
               return self;
            },
            py::is_operator() )
         .def(
            "__isub__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self -= other;
               return self;
            },
            py::is_operator() )
         .def(
            "__imul__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self *= other;
               return self;
            },
            py::is_operator() )
         .def(
            "__idiv__",
            []( VectorType& self, const VectorType& other ) -> VectorType&
            {
               self /= other;
               return self;
            },
            py::is_operator() )

         // In-place arithmetic operators (Vector OP Scalar)
         .def(
            "__iadd__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self += scalar;
               return self;
            },
            py::is_operator() )
         .def(
            "__isub__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self -= scalar;
               return self;
            },
            py::is_operator() )
         .def(
            "__imul__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self *= scalar;
               return self;
            },
            py::is_operator() )
         .def(
            "__idiv__",
            []( VectorType& self, RealType scalar ) -> VectorType&
            {
               self /= scalar;
               return self;
            },
            py::is_operator() )

         // Binary arithmetic operators (Vector OP Vector)
         .def(
            "__add__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self + other );
            },
            py::is_operator() )
         .def(
            "__sub__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self - other );
            },
            py::is_operator() )
         .def(
            "__mul__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self * other );
            },
            py::is_operator() )
         .def(
            "__truediv__",
            []( const VectorType& self, const VectorType& other )
            {
               return VectorType( self / other );
            },
            py::is_operator() )

         // Binary arithmetic operators (Vector OP Scalar)
         .def(
            "__add__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self + scalar );
            },
            py::is_operator() )
         .def(
            "__sub__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self - scalar );
            },
            py::is_operator() )
         .def(
            "__mul__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self * scalar );
            },
            py::is_operator() )
         .def(
            "__truediv__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( self / scalar );
            },
            py::is_operator() )

         // Reverse arithmetic operators (Scalar OP Vector)
         .def(
            "__radd__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar + self );
            },
            py::is_operator() )
         .def(
            "__rsub__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar - self );
            },
            py::is_operator() )
         .def(
            "__rmul__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar * self );
            },
            py::is_operator() )
         .def(
            "__rtruediv__",
            []( const VectorType& self, RealType scalar )
            {
               return VectorType( scalar / self );
            },
            py::is_operator() )

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

   // Additional operators defined only for integral value types
   if constexpr( std::is_integral_v< ValueType > ) {
      vector  //
         .def( py::self %= py::self )
         .def( py::self %= RealType() )
         .def( py::self % py::self )
         .def( py::self % RealType() )
         .def( RealType() % py::self );
   }

   // x, y, z properties
   vector.def_property(
      "x",
      []( const VectorType& self )
      {
         return self.x();
      },
      []( VectorType& self, ValueType value )
      {
         self.x() = value;
      } );
   if constexpr( VectorType::getSize() >= 2 ) {
      vector.def_property(
         "y",
         []( const VectorType& self )
         {
            return self.y();
         },
         []( VectorType& self, ValueType value )
         {
            self.y() = value;
         } );
   }
   if constexpr( VectorType::getSize() >= 3 ) {
      vector.def_property(
         "z",
         []( const VectorType& self )
         {
            return self.z();
         },
         []( VectorType& self, ValueType value )
         {
            self.z() = value;
         } );
   }

   tnl_indexing< VectorType >( vector );
}

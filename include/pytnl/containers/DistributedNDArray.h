#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Containers/DistributedNDArray.h>

template< typename Index >
void
distributed_ndarray_check_index( std::size_t i, Index idx, Index begin, Index end, Index overlap )
{
   if( idx < begin - overlap )
      throw nb::index_error( ( std::to_string( i ) + "-th index is out-of-bounds: " + std::to_string( idx ) + " < "
                               + std::to_string( begin - overlap ) )
                                .c_str() );
   if( idx >= end + overlap )
      throw nb::index_error( ( std::to_string( i ) + "-th index is out-of-bounds: " + std::to_string( idx )
                               + " >= " + std::to_string( end + overlap ) )
                                .c_str() );
}

template< typename ArrayType, typename... Args >
void
distributed_ndarray_indexing( nb::class_< ArrayType, Args... >& array )
{
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;
   constexpr std::size_t dim = ArrayType::getDimension();

   array.def(
      "__getitem__",
      [ dim ]( ArrayType& self, nb::object indices ) -> ValueType
      {
         nb::tuple tuple_indices;

         if( nb::isinstance< nb::tuple >( indices ) ) {
            tuple_indices = nb::cast< nb::tuple >( indices );
         }
         else {
            tuple_indices = nb::make_tuple( indices );
         }

         if( tuple_indices.size() != dim ) {
            throw nb::value_error( ( "Expected " + std::to_string( dim ) + " indices" ).c_str() );
         }

         std::array< IndexType, dim > indices_array;
         for( std::size_t i = 0; i < dim; ++i ) {
            indices_array[ i ] = nb::cast< IndexType >( tuple_indices[ i ] );
            distributed_ndarray_check_index(
               i, indices_array[ i ], self.getLocalBegins()[ i ], self.getLocalEnds()[ i ], self.getOverlaps()[ i ] );
         }

         // Check if the local array is allocated
         if( self.getLocalStorageSize() == 0 ) {
            throw nb::index_error( "Local storage for DistributedNDArray not allocated" );
         }

         // Unpack the array into the operator()
         return std::apply(
            [ & ]( auto... indices ) -> ValueType
            {
               // getElement is equivalent to operator[] on host but works on cuda
               return self.getElement( indices... );
            },
            indices_array );
      },
      nb::arg( "indices" ) );

   array.def(
      "__setitem__",
      [ dim ]( ArrayType& self, nb::object indices, ValueType value )
      {
         if constexpr( std::is_const_v< ValueType > )
            throw nb::type_error( "Cannot set element of a read-only array" );
         else {
            nb::tuple tuple_indices;

            if( nb::isinstance< nb::tuple >( indices ) ) {
               tuple_indices = nb::cast< nb::tuple >( indices );
            }
            else {
               tuple_indices = nb::make_tuple( indices );
            }

            if( tuple_indices.size() != dim ) {
               throw nb::value_error( ( "Expected " + std::to_string( dim ) + " indices" ).c_str() );
            }

            std::array< IndexType, dim > indices_array;
            for( std::size_t i = 0; i < dim; ++i ) {
               indices_array[ i ] = nb::cast< IndexType >( tuple_indices[ i ] );
               distributed_ndarray_check_index(
                  i, indices_array[ i ], self.getLocalBegins()[ i ], self.getLocalEnds()[ i ], self.getOverlaps()[ i ] );
            }

            // Check if the local array is allocated
            if( self.getLocalStorageSize() == 0 ) {
               throw nb::index_error( "Local storage for DistributedNDArray not allocated" );
            }

            // Unpack and assign
            std::apply(
               [ & ]( auto... indices )
               {
                  // setElement is equivalent to operator[] on host but works on cuda
                  const auto idx = self.getStorageIndex( indices... );
                  self.getLocalView().getStorageArrayView().setElement( idx, value );
               },
               indices_array );
         }
      },
      nb::arg( "indices" ),
      nb::arg( "value" ) );
}

template< typename ArrayType >
void
export_DistributedNDArray( nb::module_& m, const char* name )
{
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;

   auto array =  //
      nb::class_< ArrayType >( m, name )
         // Typedefs
         .def_prop_ro_static(  //
            "ValueType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
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
               else if constexpr( TNL::is_complex_v< ValueType > ) {
                  return nb::borrow( &PyComplex_Type );
               }
               else {
                  return nb::type< ValueType >();
               }
            } )
         .def_prop_ro_static(  //
            "ViewType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
            {
               return nb::type< typename ArrayType::ViewType >();
            } )
         .def_prop_ro_static(  //
            "ConstViewType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
            {
               return nb::type< typename ArrayType::ConstViewType >();
            } )
         .def_prop_ro_static(  //
            "LocalViewType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
            {
               return nb::type< typename ArrayType::LocalViewType >();
            } )
         .def_prop_ro_static(  //
            "ConstLocalViewType",
            []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
            {
               return nb::type< typename ArrayType::ConstLocalViewType >();
            } )

         // Constructors
         .def( nb::init<>() )
         .def( nb::init< const ArrayType& >(), nb::arg( "other" ) )

         // View getters
         .def( "getView", &ArrayType::getView )
         .def( "getConstView", &ArrayType::getConstView )
         .def( "getLocalView", &ArrayType::getLocalView )
         .def( "getConstLocalView", &ArrayType::getConstLocalView )

         // Assignment
         .def( "assign",
               []( ArrayType& array, const ArrayType& other ) -> ArrayType&
               {
                  if constexpr( std::is_const_v< ValueType > )
                     throw nb::type_error( "Cannot assign into constant array" );
                  else
                     return array = other;
               } )

         // Comparison
         .def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) )
         .def( nb::self != nb::self, nb::sig( "def __ne__(self, arg: object, /) -> bool" ) )

         // String representation
         .def(
            "__str__",
            []( const ArrayType& self )
            {
               constexpr std::size_t dim = ArrayType::getDimension();
               std::ostringstream oss;
               oss << "DistributedNDArray[" << dim << ", ";
               if constexpr( std::is_same_v< ValueType, bool > ) {
                  oss << "bool";
               }
               else if constexpr( std::is_integral_v< ValueType > ) {
                  oss << "int";
               }
               else if constexpr( std::is_floating_point_v< ValueType > ) {
                  oss << "float";
               }
               else if constexpr( TNL::is_complex_v< ValueType > ) {
                  oss << "complex";
               }
               else {
                  oss << TNL::getType( ValueType{} );
               }
               oss << ", ";
               if constexpr( std::is_same_v< DeviceType, TNL::Devices::Cuda > )
                  oss << "Cuda";
               else
                  oss << "Host";
               oss << "](";
               TNL::Algorithms::staticFor< std::size_t, 0, dim >(
                  [ & ]( auto i )
                  {
                     if constexpr( i > 0 )
                        oss << ", ";
                     oss << self.template getSize< i >();
                  } );
               oss << ")";
               return oss.str();
            },
            "Returns a readable string representation of the array" )

         // Dimension getter
         .def_static( "getDimension", &ArrayType::getDimension, "Returns the dimension of the N-dimensional array, i.e. N" )

         // Accessors
         .def( "getCommunicator", &ArrayType::getCommunicator, "Returns the MPI communicator associated with the array" )
         .def( "getSizes",
               nb::overload_cast<>( &ArrayType::getSizes, nb::const_ ),
               "Returns the sizes of the **global** array (as a tuple in Python)" )
         .def( "getOverlaps",
               nb::overload_cast<>( &ArrayType::getOverlaps, nb::const_ ),
               "Returns the overlaps of the array (as a tuple in Python)" )
         .def(
            "getLocalBegins",
            []( const ArrayType& self ) -> nb::typed< nb::tuple, nb::int_, nb::ellipsis >
            {
               std::array< std::size_t, ArrayType::getDimension() > result;
               TNL::Algorithms::staticFor< std::size_t, 0, ArrayType::getDimension() >(
                  [ & ]( auto i )
                  {
                     result[ i ] = self.getLocalBegins()[ i ];
                  } );
               return nb::tuple( nb::cast( result ) );
            },
            "Returns the beginning positions of the **local** array in the global N-dimensional array "
            "(as a tuple in Python)" )
         .def( "getLocalEnds",
               &ArrayType::getLocalEnds,
               "Returns the ending positions of the **local** array in the global N-dimensional array "
               "(as a tuple in Python)" )
         .def( "getLocalStorageSize",
               &ArrayType::getLocalStorageSize,
               "Returns the size (number of elements) of the **local** array" )

         // Storage index computation
         .def(
            "getStorageIndex",
            []( const ArrayType& self, const nb::args& indices )
            {
               if( len( static_cast< const nb::tuple& >( indices ) ) != ArrayType::getDimension() ) {
                  throw nb::value_error( "Incorrect number of indices" );
               }

               std::array< IndexType, ArrayType::getDimension() > indices_array;
               for( size_t i = 0; i < indices_array.size(); ++i ) {
                  indices_array[ i ] = nb::cast< IndexType >( indices[ i ] );
                  ndarray_check_index( i, indices_array[ i ], self.getSizes()[ i ] );
               }

               return std::apply(
                  [ & ]( auto... indices )
                  {
                     return self.getStorageIndex( indices... );
                  },
                  indices_array );
            },
            nb::sig( "def getStorageIndex(self, *indices: int) -> int" ),
            "Computes the linear storage index in the **local** array from N-dimensional **global** indices" )

      //
      ;

   distributed_ndarray_indexing( array );
   //ndarray_iteration( array );

   if constexpr( TNL::IsViewType< ArrayType >::value ) {
      array  //
             // FIXME: needed for implicit conversion from NDArray, but AllocatorType is ignored
             //.def( nb::init_implicit< TNL::Containers::DistributedNDArray< TNL::Containers::NDArray< ... > >& >() )
         .def( "bind",
               []( ArrayType& self, const ArrayType& other )
               {
                  self.bind( other );
               } )
         .def( "reset",
               &ArrayType::reset,
               "Reset the array view to the empty state. There is no deallocation, it does not affect other views." )
         //
         ;
   }
   else {
      // Additional NDArray-specific methods
      array
         // Size management
         .def(
            "setSizes",
            []( ArrayType& self, const nb::args& sizes )
            {
               constexpr std::size_t dim = ArrayType::getDimension();
               using IndexType = typename ArrayType::IndexType;

               if( sizes.size() != dim ) {
                  throw nb::value_error( ( "Expected " + std::to_string( dim ) + " sizes" ).c_str() );
               }

               std::array< IndexType, dim > sizes_array;
               for( std::size_t i = 0; i < dim; ++i ) {
                  sizes_array[ i ] = nb::cast< IndexType >( sizes[ i ] );
               }

               return std::apply(
                  [ & ]( auto... sizes )
                  {
                     self.setSizes( sizes... );
                  },
                  sizes_array );
            },
            nb::arg( "sizes" ),
            nb::sig( "def setSizes(self, *sizes: int) -> None" ),
            "Set sizes of the array using a sequence of ints" )
         .def(
            "setLike",
            []( ArrayType& self, const ArrayType& other )
            {
               self.setLike( other );
            },
            nb::arg( "other" ) )
         .def( "reset",
               &ArrayType::reset,
               "Reset the array to the empty state. The current data will be deallocated, "
               "thus all pointers and views to the array elements will become invalid." )
         .def(
            "setDistribution",
            []( ArrayType& self,
                nb::typed< nb::tuple, nb::int_, nb::ellipsis > begin,
                nb::typed< nb::tuple, nb::int_, nb::ellipsis > end )
            {
               // TODO: add communicator argument
               const TNL::MPI::Comm communicator = MPI_COMM_WORLD;
               constexpr std::size_t dim = ArrayType::getDimension();

               if( begin.size() != dim ) {
                  throw nb::value_error( ( "Invalid size of the begin multi-index: expected " + std::to_string( dim ) + ", got "
                                           + std::to_string( begin.size() ) )
                                            .c_str() );
               }
               if( end.size() != dim ) {
                  throw nb::value_error( ( "Invalid size of the end multi-index: expected " + std::to_string( dim ) + ", got "
                                           + std::to_string( end.size() ) )
                                            .c_str() );
               }

               using Array = TNL::Containers::StaticArray< dim, IndexType >;
               Array begin_array;
               Array end_array;
               for( std::size_t i = 0; i < dim; ++i ) {
                  begin_array[ i ] = nb::cast< IndexType >( begin[ i ] );
                  end_array[ i ] = nb::cast< IndexType >( end[ i ] );
               }

               self.setDistribution( begin_array, end_array, communicator );
            },
            nb::arg( "begin" ),
            nb::arg( "end" ) )
         .def( "allocate", &ArrayType::allocate )

         // Fill
         .def( "setValue", &ArrayType::setValue, nb::arg( "value" ) )

         // Deepcopy support https://pybind11.readthedocs.io/en/stable/advanced/classes.html#deepcopy-support
         .def( "__copy__",
               []( const ArrayType& self )
               {
                  return ArrayType( self );
               } )
         .def(
            "__deepcopy__",
            []( const ArrayType& self, nb::typed< nb::dict, nb::str, nb::any > )
            {
               return ArrayType( self );
            },
            nb::arg( "memo" ) );
      //
      ;
   }
}

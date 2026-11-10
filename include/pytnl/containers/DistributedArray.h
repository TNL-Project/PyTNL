#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/DistributedArraySynchronizer.h>
#include <TNL/String.h>
#include <TNL/TypeInfo.h>
#include <TNL/TypeTraits.h>

#include <memory>
#include <sstream>
#include <string>

// Returns the value type name used in the Python-facing string representations
template< typename ValueType >
std::string
distributed_value_type_name()
{
   if constexpr( std::is_same_v< ValueType, bool > )
      return "bool";
   else if constexpr( std::is_integral_v< ValueType > )
      return "int";
   else if constexpr( std::is_floating_point_v< ValueType > )
      return "float";
   else if constexpr( TNL::is_complex_v< ValueType > )
      return "complex";
   else
      return TNL::getType( ValueType{} );
}

template< typename ArrayType >
void
export_DistributedArray( nb::module_& m, const char* name )
{
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using LocalRangeType = typename ArrayType::LocalRangeType;

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
         "IndexType",
         []( nb::handle ) -> nb::typed< nb::handle, nb::type_object >
         {
            if constexpr( std::is_integral_v< IndexType > ) {
               return nb::borrow( &PyLong_Type );
            }
            else {
               return nb::type< IndexType >();
            }
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

      // Distribution
      .def(
         "setDistribution",
         []( ArrayType& self, IndexType begin, IndexType end, IndexType ghosts, IndexType globalSize )
         {
            if( begin < 0 )
               throw nb::value_error( "begin must be non-negative" );
            if( end < begin )
               throw nb::value_error( "end must be greater than or equal to begin" );
            if( ghosts < 0 )
               throw nb::value_error( "ghosts must be non-negative" );
            if( ghosts % 2 != 0 )
               throw nb::value_error( "ghosts must be even because they are split symmetrically between both neighbours" );

            // TODO: add a communicator argument (no MPI_Comm type caster yet)
            const TNL::MPI::Comm communicator = MPI_COMM_WORLD;
            const LocalRangeType localRange( begin, end );
            self.setDistribution( localRange, ghosts, globalSize, communicator );

            // TNL requires a synchronizer for any operation that updates the
            // ghost elements (e.g. setValue or copyFromGlobal), so install the
            // default one matching this distribution.
            if( ghosts > 0 ) {
               using Synchronizer = TNL::Containers::DistributedArraySynchronizer< ArrayType >;
               self.setSynchronizer( std::make_shared< Synchronizer >( localRange, ghosts / 2, communicator ) );
            }
         },
         nb::arg( "begin" ),
         nb::arg( "end" ),
         nb::arg( "ghosts" ),
         nb::arg( "globalSize" ),
         "Set the local range [begin, end), the number of ghost elements and the global size of the array. "
         "The array is distributed over MPI_COMM_WORLD." )
      .def(
         "getLocalRange",
         []( const ArrayType& self )
         {
            const LocalRangeType& range = self.getLocalRange();
            return nb::make_tuple( range.getBegin(), range.getEnd() );
         },
         nb::sig( "def getLocalRange(self) -> tuple[int, int]" ),
         "Returns the local range [begin, end) of the global array (as a tuple in Python)" )
      .def( "getGhosts", &ArrayType::getGhosts, "Returns the number of ghost elements allocated on this MPI rank" )
      .def(
         "setSynchronizer",
         []( ArrayType& self, IndexType overlaps )
         {
            if( overlaps < 0 )
               throw nb::value_error( "overlaps must be non-negative" );
            if( 2 * overlaps != self.getGhosts() )
               throw nb::value_error( ( "2 * overlaps must equal the number of ghosts (" + std::to_string( self.getGhosts() )
                                        + ")" )
                                         .c_str() );

            using Synchronizer = TNL::Containers::DistributedArraySynchronizer< ArrayType >;
            // TODO: add a communicator argument (no MPI_Comm type caster yet)
            const TNL::MPI::Comm communicator = MPI_COMM_WORLD;
            self.setSynchronizer( std::make_shared< Synchronizer >( self.getLocalRange(), overlaps, communicator ) );
         },
         nb::arg( "overlaps" ),
         "Replace the default ghost-element synchronizer with one that has the given number of overlapping "
         "elements on each side. The default synchronizer is installed by setDistribution." )
      .def( "startSynchronization",
            &ArrayType::startSynchronization,
            "Start asynchronous synchronization of the ghost elements" )
      .def( "waitForSynchronization",
            &ArrayType::waitForSynchronization,
            "Wait for the asynchronous synchronization of the ghost elements to finish" )

      // Local views
      .def( "getLocalView", &ArrayType::getLocalView, "Returns a modifiable view of the local part of the array" )
      .def( "getConstLocalView",
            &ArrayType::getConstLocalView,
            "Returns a non-modifiable view of the local part of the array" )
      .def( "getLocalViewWithGhosts",
            &ArrayType::getLocalViewWithGhosts,
            "Returns a modifiable view of the local part of the array, including ghost values" )
      .def( "getConstLocalViewWithGhosts",
            &ArrayType::getConstLocalViewWithGhosts,
            "Returns a non-modifiable view of the local part of the array, including ghost values" )

      // Global-to-local data transfer
      .def( "copyFromGlobal",
            &ArrayType::copyFromGlobal,
            nb::arg( "globalArray" ),
            "Copy the local part of the global array into this distributed array" )

      // Usual Array interface
      .def( "getSize", &ArrayType::getSize, "Returns the *global* size of the array" )
      .def( "reset",
            &ArrayType::reset,
            "Reset the array to the empty state. The current data will be deallocated, "
            "thus all pointers and views to the array elements will become invalid." )
      .def( "empty", &ArrayType::empty, "Returns true if the current array size is zero" )
      .def( "setValue", &ArrayType::setValue, nb::arg( "value" ), "Set all local elements of the array to the given value" )
      .def(
         "getElement",
         []( const ArrayType& self, IndexType i )
         {
            return self.getElement( i );
         },
         nb::arg( "i" ),
         "Returns the element with the given *global* index (the index must be local for this MPI rank)" )
      .def(
         "setElement",
         []( ArrayType& self, IndexType i, ValueType value )
         {
            if( ! self.getLocalRange().isLocal( i ) )
               throw nb::index_error( ( "Global index " + std::to_string( i ) + " is not local for this MPI rank" ).c_str() );
            self.setElement( i, value );
         },
         nb::arg( "i" ),
         nb::arg( "value" ),
         "Set the element with the given *global* index (the index must be local for this MPI rank)" )
      .def(
         "__getitem__",
         []( const ArrayType& self, IndexType i )
         {
            return self.getElement( i );
         },
         nb::arg( "i" ) )
      .def(
         "__setitem__",
         []( ArrayType& self, IndexType i, ValueType value )
         {
            if( ! self.getLocalRange().isLocal( i ) )
               throw nb::index_error( ( "Global index " + std::to_string( i ) + " is not local for this MPI rank" ).c_str() );
            self.setElement( i, value );
         },
         nb::arg( "i" ),
         nb::arg( "value" ) )
      .def(
         "setLike",
         []( ArrayType& self, const ArrayType& other )
         {
            self.setLike( other );
         },
         nb::arg( "other" ),
         "Set the distribution, size and ghost elements of this array to be the same as in the given array" )

      // Assignment
      .def(
         "assign",
         []( ArrayType& self, const ArrayType& other ) -> ArrayType&
         {
            return self = other;
         },
         nb::arg( "other" ) )

      // Comparison
      .def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) )
      .def( nb::self != nb::self, nb::sig( "def __ne__(self, arg: object, /) -> bool" ) )

      // File I/O
      .def(
         "loadFromGlobalFile",
         []( ArrayType& self, const std::string& fileName, bool allowCasting )
         {
            self.loadFromGlobalFile( TNL::String( fileName ), allowCasting );
         },
         nb::arg( "fileName" ),
         nb::arg( "allowCasting" ) = false,
         "Load the distributed array from a file containing the *global* array" )

      // String representation
      .def(
         "__str__",
         []( const ArrayType& self )
         {
            const LocalRangeType& range = self.getLocalRange();
            std::ostringstream oss;
            oss << "DistributedArray[" << distributed_value_type_name< ValueType >() << ", "
                << ( std::is_same_v< DeviceType, TNL::Devices::Cuda > ? "Cuda" : "Host" ) << "](" << self.getSize()
                << ", localRange=(" << range.getBegin() << ", " << range.getEnd() << "), ghosts=" << self.getGhosts() << ")";
            return oss.str();
         },
         "Returns a readable string representation of the array" )

      // Deepcopy support https://pybind11.readthedocs.io/en/stable/advanced/classes.html#deepcopy-support
      .def(
         "__copy__",
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
}

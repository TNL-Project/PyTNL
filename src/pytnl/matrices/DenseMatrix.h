#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Containers/Vector.h>
#include <TNL/TypeTraits.h>

#include <pytnl/containers/dlpack.h>
#include <pytnl/containers/indexing.h>

template< typename RowView, typename Scope >
void
export_DenseRowView( Scope& s, const char* name )
{
   using RealType = typename RowView::RealType;
   using IndexType = typename RowView::IndexType;

   auto rowView = nb::class_< RowView >( s, name )
                     .def( "getSize", &RowView::getSize )
                     .def( "getRowIndex", &RowView::getRowIndex )
                     .def(
                        "getColumnIndex",
                        []( const RowView& row, IndexType localIdx ) -> IndexType
                        {
                           check_array_index( row.getSize(), localIdx );
                           return row.getColumnIndex( localIdx );
                        } )
                     .def(
                        "getValue",
                        []( const RowView& row, IndexType column ) -> const RealType&
                        {
                           check_array_index( row.getSize(), column );
                           return row.getValue( column );
                        },
                        nb::rv_policy::reference_internal );

   if constexpr( ! std::is_const_v< typename RowView::RealType > ) {
      rowView
         .def(
            "setValue",
            []( RowView& row, IndexType column, RealType value ) -> void
            {
               check_array_index( row.getSize(), column );
               row.setValue( column, value );
            } )
         .def(
            "setElement",
            []( RowView& row, IndexType localIdx, IndexType column, RealType value ) -> void
            {
               // For DenseRowView, localIdx is unused (only for API compatibility with sparse);
               // column is the actual memory-accessing index — validated by TNL's getGlobalIndex(column)
               check_array_index( row.getSize(), column );
               row.setElement( localIdx, column, value );
            } );
   }
}

template< typename Matrix >
void
export_DenseMatrix( nb::module_& m, const char* name )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   auto matrix =
      nb::class_< Matrix >( m, name )
         .def( nb::init<>() )
         .def( nb::init< IndexType, IndexType >() )
         // File I/O
         .def_static( "getSerializationType", &Matrix::getSerializationType )
         .def( "save", &Matrix::save )
         .def( "load", &Matrix::load )

         .def( "print", &Matrix::print )
         .def(
            "__str__",
            []( Matrix& m )
            {
               std::stringstream ss;
               ss << m;
               return ss.str();
            } )

         // Matrix
         .def( "setDimensions", nb::overload_cast< IndexType, IndexType >( &Matrix::setDimensions ) )
         .def(
            "setLike",
            []( Matrix& matrix, const Matrix& other ) -> void
            {
               matrix.setLike( other );
            } )
         .def( "getAllocatedElementsCount", &Matrix::getAllocatedElementsCount )
         .def( "getNonzeroElementsCount", &Matrix::getNonzeroElementsCount )
         .def( "reset", &Matrix::reset )
         .def( "getRows", &Matrix::getRows )
         .def( "getColumns", &Matrix::getColumns )
         .def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) )
         .def( nb::self != nb::self, nb::sig( "def __ne__(self, arg: object, /) -> bool" ) )

         // DenseMatrix
         .def( "setValue", &Matrix::setValue )
         .def( "setRowCapacities", &Matrix::template setRowCapacities< IndexVectorType > )
         .def( "getRowCapacities", &Matrix::template getRowCapacities< IndexVectorType > )
         .def( "getCompressedRowLengths", &Matrix::template getCompressedRowLengths< IndexVectorType > )
         .def( "getRowCapacity", &Matrix::getRowCapacity )
         .def(
            "getRow",
            []( Matrix& matrix, IndexType rowIdx ) -> typename Matrix::RowView
            {
               if( rowIdx < 0 || rowIdx >= matrix.getRows() )
                  throw nb::index_error( ( "row index " + std::to_string( rowIdx ) + " is out-of-bounds for matrix with "
                                           + std::to_string( matrix.getRows() ) + " rows" )
                                            .c_str() );
               return matrix.getRow( rowIdx );
            } )
         .def(
            "setElement",
            []( Matrix& matrix, IndexType row, IndexType col, RealType value )
            {
               check_matrix_index( matrix.getRows(), matrix.getColumns(), row, col );
               matrix.setElement( row, col, value );
            },
            nb::arg( "row" ),
            nb::arg( "col" ),
            nb::arg( "value" ) )
         .def(
            "addElement",
            []( Matrix& matrix, IndexType row, IndexType col, RealType value, RealType thisElementMultiplicator )
            {
               check_matrix_index( matrix.getRows(), matrix.getColumns(), row, col );
               matrix.addElement( row, col, value, thisElementMultiplicator );
            },
            nb::arg( "row" ),
            nb::arg( "col" ),
            nb::arg( "value" ),
            nb::arg( "thisElementMultiplicator" ) = 1.0 )
         .def(
            "getElement",
            []( const Matrix& matrix, IndexType row, IndexType col ) -> RealType
            {
               check_matrix_index( matrix.getRows(), matrix.getColumns(), row, col );
               return matrix.getElement( row, col );
            },
            nb::arg( "row" ),
            nb::arg( "col" ) )
         .def(
            "vectorProduct",
            []( Matrix& matrix,
                const VectorType& inVector,
                VectorType& outVector,
                RealType matrixMultiplicator = 1.0,
                RealType outVectorMultiplicator = 0.0,
                IndexType begin = 0,
                IndexType end = 0 ) -> void
            {
               matrix.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, begin, end );
            } )
         .def(
            "assign",
            []( Matrix& matrix, const Matrix& other ) -> Matrix&
            {
               return matrix = other;
            } )

         // accessors for internal vectors
         .def( "getValues", nb::overload_cast<>( &Matrix::getValues ), nb::rv_policy::reference_internal );

   if constexpr( nb::dtype< RealType >().bits != 0 ) {
      matrix
         .def(
            "__dlpack__",
            []( nb::pointer_and_handle< Matrix > self, nb::kwargs kwargs )
            {
               const IndexType rows = self.p->getRows();
               const IndexType cols = self.p->getColumns();

               std::array< std::size_t, 2 > shape{ static_cast< std::size_t >( rows ), static_cast< std::size_t >( cols ) };

               std::int64_t row_stride;
               std::int64_t col_stride;
               if constexpr( Matrix::getOrganization() == TNL::Algorithms::Segments::RowMajorOrder ) {
                  row_stride = static_cast< std::int64_t >( cols );
                  col_stride = 1;
               }
               else {
                  row_stride = 1;
                  col_stride = static_cast< std::int64_t >( rows );
               }
               std::array< std::int64_t, 2 > strides{ row_stride, col_stride };

               auto [ dl_device, device_id ] = dlpack_device< VectorType >();

               using array_api_t = nb::ndarray< nb::array_api, RealType >;
               array_api_t array_api(
                  self.p->getValues().getData(),
                  2,
                  shape.data(),
                  self.h,
                  strides.data(),
                  nb::dtype< RealType >(),
                  dl_device,
                  device_id );

               nb::object aa = nb::cast( array_api, nb::rv_policy::reference_internal, self.h );
               return aa.attr( "__dlpack__" )( **kwargs );
            },
            nb::sig( "def __dlpack__(self, **kwargs: typing.Any) -> typing_extensions.CapsuleType" ) )
         .def_static( "__dlpack_device__", dlpack_device< VectorType > );
   }
}

#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/TypeTraits.h>

#include <pytnl/containers/dlpack.h>
#include <pytnl/containers/indexing.h>
#include "MatrixBase.h"

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

// Adds dense-specific methods to an already-created nb::class_ for
// DenseMatrixBase. Called by export_DenseMatrixBaseClass.
// MatrixBase-level methods (getRows, getColumns, getAllocatedElementsCount,
// isBinary, isSymmetric) are added via def_MatrixBaseMethods.
template< typename Matrix, typename MatrixClass >
void
def_DenseMatrixBaseMethods( MatrixClass& matrix )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   // RealType has const stripped (MatrixBase uses std::remove_const_t),
   // so we check the ValuesViewType for the actual const-ness
   using ValueType = typename Matrix::ValuesViewType::ValueType;

   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   matrix.def( "print", &Matrix::print )
      .def(
         "__str__",
         []( Matrix& m )
         {
            std::stringstream ss;
            ss << m;
            return ss.str();
         } )
      .def( "getNonzeroElementsCount", &Matrix::getNonzeroElementsCount )
      .def( "getRowCapacities", &Matrix::template getRowCapacities< IndexVectorType > )
      .def( "getCompressedRowLengths", &Matrix::template getCompressedRowLengths< IndexVectorType > )
      .def( "getRowCapacity", &Matrix::getRowCapacity )
      .def(
         "getElement",
         []( const Matrix& m, IndexType row, IndexType col ) -> RealType
         {
            check_matrix_index( m.getRows(), m.getColumns(), row, col );
            return m.getElement( row, col );
         },
         nb::arg( "row" ),
         nb::arg( "col" ) )
      .def( "getValues", nb::overload_cast<>( &Matrix::getValues ), nb::rv_policy::reference_internal );

   if constexpr( nb::dtype< ValueType >().bits != 0 ) {
      using ValuesViewType = typename Matrix::ValuesViewType;
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

               auto [ dl_device, device_id ] = dlpack_device< ValuesViewType >();

               using array_api_t = nb::ndarray< nb::array_api, ValueType >;
               array_api_t array_api(
                  self.p->getValues().getData(),
                  2,
                  shape.data(),
                  self.h,
                  strides.data(),
                  nb::dtype< ValueType >(),
                  dl_device,
                  device_id );

               nb::object aa = nb::cast( array_api, nb::rv_policy::reference_internal, self.h );
               return aa.attr( "__dlpack__" )( **kwargs );
            },
            nb::sig( "def __dlpack__(self, **kwargs: typing.Any) -> typing_extensions.CapsuleType" ) )
         .def_static( "__dlpack_device__", dlpack_device< ValuesViewType > );
   }

   if constexpr( ! std::is_const_v< ValueType > ) {
      using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
      matrix.def( "setValue", &Matrix::setValue )
         .def(
            "getRow",
            []( Matrix& m, IndexType rowIdx ) -> typename Matrix::RowView
            {
               if( rowIdx < 0 || rowIdx >= m.getRows() )
                  throw nb::index_error( ( "row index " + std::to_string( rowIdx ) + " is out-of-bounds for matrix with "
                                           + std::to_string( m.getRows() ) + " rows" )
                                            .c_str() );
               return m.getRow( rowIdx );
            } )
         .def(
            "setElement",
            []( Matrix& m, IndexType row, IndexType col, RealType value )
            {
               check_matrix_index( m.getRows(), m.getColumns(), row, col );
               m.setElement( row, col, value );
            },
            nb::arg( "row" ),
            nb::arg( "col" ),
            nb::arg( "value" ) )
         .def(
            "addElement",
            []( Matrix& m, IndexType row, IndexType col, RealType value, RealType thisElementMultiplicator )
            {
               check_matrix_index( m.getRows(), m.getColumns(), row, col );
               m.addElement( row, col, value, thisElementMultiplicator );
            },
            nb::arg( "row" ),
            nb::arg( "col" ),
            nb::arg( "value" ),
            nb::arg( "thisElementMultiplicator" ) = 1.0 )
         .def(
            "vectorProduct",
            []( Matrix& m,
                const VectorType& inVector,
                VectorType& outVector,
                RealType matrixMultiplicator = 1.0,
                RealType outVectorMultiplicator = 0.0,
                IndexType begin = 0,
                IndexType end = 0 ) -> void
            {
               m.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, begin, end );
            } );
   }
}

// MatrixBase-level methods are added via def_MatrixBaseMethods since
// MatrixBase cannot be a nanobind base class (Organization differs per format).
template< typename DenseMatrixBaseType >
void
export_DenseMatrixBaseClass( nb::module_& m, const char* name )
{
   auto base = nb::class_< DenseMatrixBaseType >( m, name );
   def_MatrixBaseMethods< DenseMatrixBaseType >( base );
   def_DenseMatrixBaseMethods< DenseMatrixBaseType >( base );
}

template< typename Matrix, typename BaseType >
void
export_DenseMatrix( nb::module_& m, const char* name )
{
   using IndexType = typename Matrix::IndexType;

   auto matrix = nb::class_< Matrix, BaseType >( m, name )
                    .def( nb::init<>() )
                    .def( nb::init< IndexType, IndexType >() )
                    // File I/O
                    .def_static( "getSerializationType", &Matrix::getSerializationType )
                    .def( "save", &Matrix::save )
                    .def( "load", &Matrix::load )

                    // Matrix
                    .def( "setDimensions", nb::overload_cast< IndexType, IndexType >( &Matrix::setDimensions ) )
                    .def( "reset", &Matrix::reset )
                    .def(
                       "setLike",
                       []( Matrix& m, const Matrix& other ) -> void
                       {
                          m.setLike( other );
                       } )

                    // DenseMatrix
                    .def(
                       "assign",
                       []( Matrix& m, const Matrix& other ) -> Matrix&
                       {
                          return m = other;
                       } )

                    // Views
                    .def( "getView", &Matrix::getView, nb::rv_policy::reference_internal )
                    .def( "getConstView", &Matrix::getConstView, nb::rv_policy::reference_internal )

                    // operator== calls getConstView() which exists only on owning matrices
                    .def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) )
                    .def( nb::self != nb::self, nb::sig( "def __ne__(self, arg: object, /) -> bool" ) );
}

template< typename ViewType, typename BaseType >
void
export_DenseMatrixView( nb::module_& m, const char* name )
{
   auto view = nb::class_< ViewType, BaseType >( m, name );

   // Copy constructor
   view.def( nb::init< const ViewType& >() );

   view.def(
      "bind",
      []( ViewType& self, ViewType& other )
      {
         self.bind( other );
      },
      nb::keep_alive< 1, 2 >(),
      "Bind this view to another view.\n\n"
      "Warning: TNL views are non-owning. The source view's parent matrix\n"
      "must outlive this view — if the parent is garbage-collected,\n"
      "accessing this view results in undefined behavior." );

   // operator== is on DenseMatrixBase but calls getConstView() which exists
   // on ViewType, so it must be registered here (not on the base class)
   view.def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) );
   view.def( nb::self != nb::self, nb::sig( "def __ne__(self, arg: object, /) -> bool" ) );
}

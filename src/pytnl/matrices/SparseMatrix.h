#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Containers/Vector.h>
#include <TNL/TypeTraits.h>

#include <pytnl/containers/indexing.h>

template< typename RowView, typename Scope >
void
export_RowView( Scope& s, const char* name )
{
   using RealType = typename RowView::RealType;
   using IndexType = typename RowView::IndexType;

   auto rowView = nb::class_< RowView >( s, name )
                     .def( "getSize", &RowView::getSize )
                     .def( "getRowIndex", &RowView::getRowIndex )
                     .def(
                        "getColumnIndex",
                        []( const RowView& row, IndexType localIdx ) -> const IndexType&
                        {
                           check_array_index( row.getSize(), localIdx );
                           return row.getColumnIndex( localIdx );
                        },
                        nb::rv_policy::reference_internal )
                     .def(
                        "getValue",
                        []( const RowView& row, IndexType localIdx ) -> const RealType&
                        {
                           check_array_index( row.getSize(), localIdx );
                           return row.getValue( localIdx );
                        },
                        nb::rv_policy::reference_internal )
                     .def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) );

   if constexpr( ! std::is_const_v< typename RowView::RealType > ) {
      rowView
         .def(
            "setValue",
            []( RowView& row, IndexType localIdx, RealType value )
            {
               check_array_index( row.getSize(), localIdx );
               row.setValue( localIdx, value );
            } )
         .def(
            "setColumnIndex",
            []( RowView& row, IndexType localIdx, IndexType colIdx )
            {
               check_array_index( row.getSize(), localIdx );
               row.setColumnIndex( localIdx, colIdx );
            } )
         .def(
            "setElement",
            []( RowView& row, IndexType localIdx, IndexType colIdx, RealType value )
            {
               check_array_index( row.getSize(), localIdx );
               row.setElement( localIdx, colIdx, value );
            } );
   }
}

template< typename Segments, typename Enable = void >
struct export_CSR
{
   template< typename Scope >
   static void
   e( Scope& s )
   {}
};

template< typename Segments >
struct export_CSR< Segments, typename TNL::enable_if_type< decltype( Segments{}.getOffsets() ) >::type >
{
   template< typename Scope >
   static void
   e( Scope& s )
   {
      s.def(
         "getOffsets",
         []( const Segments& segments ) -> typename Segments::ConstOffsetsView
         {
            return segments.getOffsets();
         },
         nb::rv_policy::reference_internal );
   }
};

template< typename Segments, typename Scope >
void
export_Segments( Scope& s, const char* name )
{
   // getSegments() returns a reference to SegmentsViewType, not SegmentsType,
   // so we must register the ViewType to make the return value convertible
   using SegmentsView = typename Segments::ViewType;

   auto segments =
      nb::class_< SegmentsView >( s, name )
         .def( "getSegmentCount", &SegmentsView::getSegmentCount )
         // Ellpack has a getSegmentSize overload without arguments
         .def(
            "getSegmentSize",
            []( const SegmentsView& segments, typename SegmentsView::IndexType segmentIdx ) -> typename SegmentsView::IndexType
            {
               return segments.getSegmentSize( segmentIdx );
            } )
         .def( "getElementCount", &SegmentsView::getElementCount )
         .def( "getStorageSize", &SegmentsView::getStorageSize )
         .def( "getGlobalIndex", &SegmentsView::getGlobalIndex )
      // FIXME: this does not compile
      //      .def(nb::self == nb::self)
      // TODO: forElements, forAllElements, forSegments, forAllSegments,
      // segmentsReduction, allReduction
      ;

   export_CSR< SegmentsView >::e( segments );
}

template< typename Matrix >
void
export_Matrix( nb::module_& m, const char* name )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using ComputeRealType = typename Matrix::ComputeRealType;

   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   auto matrix =
      nb::class_< Matrix >( m, name )
         .def( nb::init<>() )
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
         // TODO: export for more types
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
         // TODO: export for more types
         .def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) )
         .def( nb::self != nb::self, nb::sig( "def __ne__(self, arg: object, /) -> bool" ) )

         // SparseMatrix
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
            []( Matrix& matrix, IndexType row, IndexType col, RealType value, ComputeRealType thisElementMultiplicator )
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
         // TODO: reduceRows, reduceAllRows, forElements, forAllElements,
         // forRows, forAllRows
         // TODO: export for more types
         .def(
            "vectorProduct",
            []( Matrix& matrix,
                const VectorType& inVector,
                VectorType& outVector,
                ComputeRealType matrixMultiplicator = 1.0,
                ComputeRealType outVectorMultiplicator = 0.0,
                IndexType begin = 0,
                IndexType end = 0 ) -> void
            {
               matrix.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, begin, end );
            } )
         // TODO: these two don't work
         //.def("addMatrix",           &Matrix::addMatrix)
         //.def("getTransposition",    &Matrix::getTransposition)
         // TODO: export for more types
         .def(
            "assign",
            []( Matrix& matrix, const Matrix& other ) -> Matrix&
            {
               return matrix = other;
            } )

         // accessors for internal vectors
         .def( "getValues", nb::overload_cast<>( &Matrix::getValues ), nb::rv_policy::reference_internal )
         .def( "getColumnIndexes", nb::overload_cast<>( &Matrix::getColumnIndexes ), nb::rv_policy::reference_internal )
         .def( "getSegments", nb::overload_cast<>( &Matrix::getSegments ), nb::rv_policy::reference_internal );

   export_Segments< typename Matrix::SegmentsType >( matrix, "Segments" );
}

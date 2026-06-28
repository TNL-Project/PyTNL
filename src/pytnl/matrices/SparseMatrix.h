#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/SparseMatrixView.h>
#include <TNL/TypeTraits.h>

#include <pytnl/containers/indexing.h>
#include "MatrixBase.h"

template< typename RowView, typename Scope >
void
export_RowView( Scope& s, const char* name )
{
   using RealType = typename RowView::RealType;
   using IndexType = typename RowView::IndexType;

   auto rowView = nb::class_< RowView >( s, name )
                     .def( "getSize", &RowView::getSize )
                     .def( "getRowIndex", &RowView::getRowIndex )
                     .def_static( "isBinary", &RowView::isBinary )
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
                     .def(
                        "getGlobalIndex",
                        []( const RowView& row, IndexType localIdx ) -> IndexType
                        {
                           check_array_index( row.getSize(), localIdx );
                           return row.getGlobalIndex( localIdx );
                        } )
                     .def(
                        "__str__",
                        []( RowView& row )
                        {
                           std::stringstream ss;
                           ss << row;
                           return ss.str();
                        } )
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
            } )
         .def( "sortColumnIndexes", &RowView::sortColumnIndexes );
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

// Adds sparse-specific methods to an already-created nb::class_ for
// SparseMatrixBase. Called by export_SparseMatrixBaseClass.
// MatrixBase-level methods (getRows, getColumns, getAllocatedElementsCount,
// isBinary, isSymmetric) are added via def_MatrixBaseMethods.
template< typename Matrix, typename MatrixClass >
void
def_SparseMatrixBaseMethods( MatrixClass& matrix )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using ComputeRealType = typename Matrix::ComputeRealType;

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
      .def( "getValues", nb::overload_cast<>( &Matrix::getValues ), nb::rv_policy::reference_internal )
      .def( "getColumnIndexes", nb::overload_cast<>( &Matrix::getColumnIndexes ), nb::rv_policy::reference_internal )
      .def( "getSegments", nb::overload_cast<>( &Matrix::getSegments ), nb::rv_policy::reference_internal );

   if constexpr( ! std::is_const_v< ValueType > ) {
      using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
      matrix
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
            []( Matrix& m, IndexType row, IndexType col, RealType value, ComputeRealType thisElementMultiplicator )
            {
               check_matrix_index( m.getRows(), m.getColumns(), row, col );
               m.addElement( row, col, value, thisElementMultiplicator );
            },
            nb::arg( "row" ),
            nb::arg( "col" ),
            nb::arg( "value" ),
            nb::arg( "thisElementMultiplicator" ) = 1.0 )
         // TODO: reduceRows, reduceAllRows, forElements, forAllElements,
         // forRows, forAllRows
         // TODO: export for more types
         .def(
            "vectorProduct",
            []( Matrix& m,
                const VectorType& inVector,
                VectorType& outVector,
                ComputeRealType matrixMultiplicator = 1.0,
                ComputeRealType outVectorMultiplicator = 0.0,
                IndexType begin = 0,
                IndexType end = 0 ) -> void
            {
               m.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, begin, end );
            } );
   }
}

// MatrixBase-level methods are added via def_MatrixBaseMethods since
// MatrixBase cannot be a nanobind base class (Organization differs per format).
template< typename SparseMatrixBaseType >
void
export_SparseMatrixBaseClass( nb::module_& m, const char* name )
{
   auto base = nb::class_< SparseMatrixBaseType >( m, name );
   def_MatrixBaseMethods< SparseMatrixBaseType >( base );
   def_SparseMatrixBaseMethods< SparseMatrixBaseType >( base );
}

template< typename Matrix, typename BaseType >
void
export_Matrix( nb::module_& m, const char* name )
{
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   auto matrix = nb::class_< Matrix, BaseType >( m, name )
                    .def( nb::init<>() )
                    // File I/O
                    .def_static( "getSerializationType", &Matrix::getSerializationType )
                    .def( "save", &Matrix::save )
                    .def( "load", &Matrix::load )

                    // Matrix
                    .def( "setDimensions", nb::overload_cast< IndexType, IndexType >( &Matrix::setDimensions ) )
                    .def( "reset", &Matrix::reset )
                    // TODO: export for more types
                    .def(
                       "setLike",
                       []( Matrix& m, const Matrix& other ) -> void
                       {
                          m.setLike( other );
                       } )

                    // SparseMatrix
                    .def( "setRowCapacities", &Matrix::template setRowCapacities< IndexVectorType > )
                    // TODO: these two don't work
                    //.def("addMatrix",           &Matrix::addMatrix)
                    //.def("getTransposition",    &Matrix::getTransposition)
                    // TODO: export for more types
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

   export_Segments< typename Matrix::SegmentsType >( matrix, "Segments" );
}

template< typename ViewType, typename BaseType >
void
export_SparseMatrixView( nb::module_& m, const char* name )
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

   // operator== is on SparseMatrixBase but calls getConstView() which exists
   // on ViewType, so it must be registered here (not on the base class)
   view.def( nb::self == nb::self, nb::sig( "def __eq__(self, arg: object, /) -> bool" ) );
   view.def( nb::self != nb::self, nb::sig( "def __ne__(self, arg: object, /) -> bool" ) );
}

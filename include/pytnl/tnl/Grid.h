#pragma once

#include <type_traits>

#include <pytnl/nanobind.h>

#include "Grid_getSpaceStepsProducts.h"
#include "StaticVector.h"
#include "mesh_getters.h"

template< typename GridEntity, typename PyGrid >
void
export_GridEntity( PyGrid& scope, const char* name )
{
   typename GridEntity::CoordinatesType const& ( GridEntity::*_getCoordinates1 )( void ) const = &GridEntity::getCoordinates;
   typename GridEntity::CoordinatesType& ( GridEntity::*_getCoordinates2 )( void ) = &GridEntity::getCoordinates;

   nb::class_< GridEntity >( scope, name )
      .def( nb::init< typename GridEntity::GridType >(), nb::rv_policy::reference_internal )
      .def( nb::init< typename GridEntity::GridType, typename GridEntity::CoordinatesType >(),
            nb::rv_policy::reference_internal )
      .def(
         nb::init< typename GridEntity::GridType, typename GridEntity::CoordinatesType, typename GridEntity::CoordinatesType >(),
         nb::rv_policy::reference_internal )
      .def( nb::init< typename GridEntity::GridType,
                      typename GridEntity::CoordinatesType,
                      typename GridEntity::CoordinatesType,
                      typename GridEntity::IndexType >(),
            nb::rv_policy::reference_internal )
      .def( nb::init< typename GridEntity::GridType, typename GridEntity::IndexType >(), nb::rv_policy::reference_internal )
      .def( "getEntityDimension", &GridEntity::getEntityDimension )
      .def( "getMeshDimension", &GridEntity::getMeshDimension )
      // TODO: constructors
      .def( "getCoordinates", _getCoordinates1, nb::rv_policy::reference_internal )
      .def( "getCoordinates", _getCoordinates2, nb::rv_policy::reference_internal )
      .def( "setCoordinates", &GridEntity::setCoordinates )
      .def( "refresh", &GridEntity::refresh )
      .def( "getIndex", &GridEntity::getIndex )
      .def( "getNormals", &GridEntity::getNormals )
      .def( "setNormals", &GridEntity::setNormals )
      .def( "getOrientation", &GridEntity::getOrientation )
      .def( "setOrientation", &GridEntity::setOrientation )
      .def( "getBasis", &GridEntity::getBasis )
      // TODO: getNeighbourEntity
      .def( "isBoundary", &GridEntity::isBoundary )
      .def( "getCenter", &GridEntity::getCenter )
      .def( "getMeasure", &GridEntity::getMeasure )
      .def( "getMesh", &GridEntity::getMesh, nb::rv_policy::reference_internal );
}

template< typename Grid >
void
export_Grid( nb::module_& m, const char* name )
{
   void ( Grid::*_setDimensions )( const typename Grid::CoordinatesType& ) = &Grid::setDimensions;
   void ( Grid::*_setOrigin )( const typename Grid::PointType& ) = &Grid::setOrigin;

   auto grid =
      nb::class_< Grid >( m, name )
         .def( nb::init<>() )
         .def_static( "getMeshDimension", &Grid::getMeshDimension )
         .def( "setDimensions", _setDimensions )
         .def( "getDimensions", &Grid::getDimensions, nb::rv_policy::reference_internal )
         .def( "setDomain", &Grid::setDomain )
         .def( "setOrigin", _setOrigin )
         .def( "getOrigin", &Grid::getOrigin, nb::rv_policy::reference_internal )
         .def( "getProportions", &Grid::getProportions, nb::rv_policy::reference_internal )
         // NOTE: if combined into getEntity, the return type would depend on
         // the runtime parameter (entity)
         .def( "getCell",
               nb::overload_cast< typename Grid::IndexType >( &Grid::template getEntity< typename Grid::Cell >, nb::const_ ) )
         .def( "getCell",
               nb::overload_cast< const typename Grid::CoordinatesType& >( &Grid::template getEntity< typename Grid::Cell >,
                                                                           nb::const_ ) )
         .def( "getFace",
               nb::overload_cast< typename Grid::IndexType >( &Grid::template getEntity< typename Grid::Face >, nb::const_ ) )
         .def( "getFace",
               nb::overload_cast< const typename Grid::CoordinatesType& >( &Grid::template getEntity< typename Grid::Face >,
                                                                           nb::const_ ) )
         .def( "getVertex",
               nb::overload_cast< typename Grid::IndexType >( &Grid::template getEntity< typename Grid::Vertex >, nb::const_ ) )
         .def( "getVertex",
               nb::overload_cast< const typename Grid::CoordinatesType& >( &Grid::template getEntity< typename Grid::Vertex >,
                                                                           nb::const_ ) )
         .def( "getEntityIndex", &Grid::template getEntityIndex< typename Grid::Cell > )
         .def( "getEntityIndex", &Grid::template getEntityIndex< typename Grid::Face > )
         .def( "getEntityIndex", &Grid::template getEntityIndex< typename Grid::Vertex > )
         .def( "getCellMeasure", &Grid::getCellMeasure, nb::rv_policy::reference_internal )
         .def( "getSpaceSteps", &Grid::getSpaceSteps, nb::rv_policy::reference_internal )
         .def( "getSmallestSpaceStep", &Grid::getSmallestSpaceStep )

         // Comparison operators
         .def( nb::self == nb::self )
         .def( nb::self != nb::self )

      // TODO: more?
      ;

   // complicated methods
   export_getEntitiesCount( grid );
   SpaceStepsProductsGetter< Grid >::export_getSpaceSteps( grid );

   // nested types
   export_StaticVector< typename Grid::CoordinatesType >( grid, "CoordinatesType" );
   export_StaticVector< typename Grid::PointType >( grid, "PointType" );
   export_GridEntity< typename Grid::Cell >( grid, "Cell" );
   export_GridEntity< typename Grid::Face >( grid, "Face" );
   // avoid duplicate conversion if the type is the same
   if( ! std::is_same< typename Grid::Face, typename Grid::Vertex >::value )
      export_GridEntity< typename Grid::Vertex >( grid, "Vertex" );
}

#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "StaticVector.h"
#include "Grid_getSpaceStepsProducts.h"
#include "mesh_getters.h"

#include <type_traits>

template< typename GridEntity, typename PyGrid >
void export_GridEntity( PyGrid & scope, const char* name )
{
    typename GridEntity::CoordinatesType const & (GridEntity::* _getCoordinates1)(void) const = &GridEntity::getCoordinates;
    typename GridEntity::CoordinatesType       & (GridEntity::* _getCoordinates2)(void) = &GridEntity::getCoordinates;

    py::class_< GridEntity >( scope, name )
        .def( py::init< typename GridEntity::GridType >(),
              py::return_value_policy::reference_internal )
        .def( py::init< typename GridEntity::GridType,
                        typename GridEntity::CoordinatesType >(),
              py::return_value_policy::reference_internal )
        .def( py::init< typename GridEntity::GridType,
                        typename GridEntity::CoordinatesType,
                        typename GridEntity::CoordinatesType >(),
              py::return_value_policy::reference_internal )
        .def( py::init< typename GridEntity::GridType,
                        typename GridEntity::CoordinatesType,
                        typename GridEntity::CoordinatesType,
                        typename GridEntity::IndexType >(),
              py::return_value_policy::reference_internal )
        .def( py::init< typename GridEntity::GridType,
                        typename GridEntity::IndexType >(),
              py::return_value_policy::reference_internal )
        .def("getEntityDimension", &GridEntity::getEntityDimension)
        .def("getMeshDimension", &GridEntity::getMeshDimension)
        // TODO: constructors
        .def("getCoordinates", _getCoordinates1, py::return_value_policy::reference_internal)
        .def("getCoordinates", _getCoordinates2, py::return_value_policy::reference_internal)
        .def("setCoordinates", &GridEntity::setCoordinates)
        .def("refresh", &GridEntity::refresh)
        .def("getIndex", &GridEntity::getIndex)
        .def("getNormals", &GridEntity::getNormals)
        .def("setNormals", &GridEntity::setNormals)
        .def("getOrientation", &GridEntity::getOrientation)
        .def("setOrientation", &GridEntity::setOrientation)
        .def("getBasis", &GridEntity::getBasis)
        // TODO: getNeighbourEntity
        .def("isBoundary", &GridEntity::isBoundary)
        .def("getCenter", &GridEntity::getCenter)
        .def("getMeasure", &GridEntity::getMeasure)
        .def("getMesh", &GridEntity::getMesh, py::return_value_policy::reference_internal)
    ;
}

template< typename Grid >
void export_Grid( py::module & m, const char* name )
{
    void (Grid::* _setDimensions)(const typename Grid::CoordinatesType &) = &Grid::setDimensions;
    void (Grid::* _setOrigin)(const typename Grid::PointType&) = &Grid::setOrigin;

    auto grid = py::class_<Grid>( m, name )
        .def(py::init<>())
        .def_static("getMeshDimension", &Grid::getMeshDimension)
        .def("setDimensions", _setDimensions)
        .def("getDimensions", &Grid::getDimensions, py::return_value_policy::reference_internal)
        .def("setDomain", &Grid::setDomain)
        .def("setOrigin", _setOrigin)
        .def("getOrigin", &Grid::getOrigin, py::return_value_policy::reference_internal)
        .def("getProportions", &Grid::getProportions, py::return_value_policy::reference_internal)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Grid, typename Grid::Cell >)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Grid, typename Grid::Face >)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Grid, typename Grid::Vertex >)
        // NOTE: if combined into getEntity, the return type would depend on the runtime parameter (entity)
        .def("getCell", py::overload_cast< typename Grid::IndexType >( &Grid::template getEntity<typename Grid::Cell>, py::const_ ))
        .def("getCell", py::overload_cast< const typename Grid::CoordinatesType& >( &Grid::template getEntity<typename Grid::Cell>, py::const_ ))
        .def("getFace", py::overload_cast< typename Grid::IndexType >( &Grid::template getEntity<typename Grid::Face>, py::const_ ))
        .def("getFace", py::overload_cast< const typename Grid::CoordinatesType& >( &Grid::template getEntity<typename Grid::Face>, py::const_ ))
        .def("getVertex", py::overload_cast< typename Grid::IndexType >( &Grid::template getEntity<typename Grid::Vertex>, py::const_ ))
        .def("getVertex", py::overload_cast< const typename Grid::CoordinatesType& >( &Grid::template getEntity<typename Grid::Vertex>, py::const_ ))
        .def("getEntityIndex", &Grid::template getEntityIndex<typename Grid::Cell>)
        .def("getEntityIndex", &Grid::template getEntityIndex<typename Grid::Face>)
        .def("getEntityIndex", &Grid::template getEntityIndex<typename Grid::Vertex>)
        .def("getCellMeasure", &Grid::getCellMeasure, py::return_value_policy::reference_internal)
        .def("getSpaceSteps", &Grid::getSpaceSteps, py::return_value_policy::reference_internal)
        .def("getSmallestSpaceStep", &Grid::getSmallestSpaceStep)
    ;

    // complicated methods
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

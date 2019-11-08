#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "StaticVector.h"
#include "Grid_getSpaceStepsProducts.h"
#include "EntityTypes.h"

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
                        typename GridEntity::CoordinatesType,
                        typename GridEntity::EntityOrientationType,
                        typename GridEntity::EntityBasisType >(),
              py::return_value_policy::reference_internal )
        .def("getEntityDimension", &GridEntity::getEntityDimension)
        .def("getMeshDimension", &GridEntity::getMeshDimension)
        // TODO: constructors
        .def("getCoordinates", _getCoordinates1, py::return_value_policy::reference_internal)
        .def("getCoordinates", _getCoordinates2, py::return_value_policy::reference_internal)
        .def("setCoordinates", &GridEntity::setCoordinates)
        .def("refresh", &GridEntity::refresh)
        .def("getIndex", &GridEntity::getIndex)
        // FIXME: some templates return reference, others value
//        .def("getOrientation", &GridEntity::getOrientation, py::return_internal_reference<>(), py::return_value_policy<py::copy_const_reference>())
        .def("setOrientation", &GridEntity::setOrientation)
        // FIXME: some templates return reference, others value
//        .def("getBasis", &GridEntity::getBasis, py::return_internal_reference<>(), py::return_value_policy<py::copy_const_reference>())
        .def("setBasis", &GridEntity::setBasis)
        // TODO: getNeighbourEntities
        .def("isBoundaryEntity", &GridEntity::isBoundaryEntity)
        .def("getCenter", &GridEntity::getCenter)
        // FIXME: some templates return reference, others value
//        .def("getMeasure", &GridEntity::getMeasure, py::return_value_policy<py::copy_const_reference>())
        .def("getMesh", &GridEntity::getMesh, py::return_value_policy::reference_internal)
    ;
}

template< typename Grid >
void export_Grid( py::module & m, const char* name )
{
    // function pointers for overloaded methods
// FIXME: number of parameters depends on the grid dimension
//    void (Grid::* _setDimensions1)(const IndexType) = &Grid::setDimensions;
    void (Grid::* _setDimensions2)(const typename Grid::CoordinatesType &) = &Grid::setDimensions;

    export_EntityTypes(m);

    auto grid = py::class_<Grid, TNL::Object>( m, name )
        .def(py::init<>())
        .def_static("getMeshDimension", &Grid::getMeshDimension)
        .def_static("getSerializationType", &Grid::getSerializationType)
        .def("getSerializationTypeVirtual", &Grid::getSerializationTypeVirtual)
        // FIXME: number of parameters depends on the grid dimension
//        .def("setDimensions", _setDimensions1)
        .def("setDimensions", _setDimensions2)
        .def("getDimensions", &Grid::getDimensions, py::return_value_policy::reference_internal)
        .def("setDomain", &Grid::setDomain)
        .def("getOrigin", &Grid::getOrigin, py::return_value_policy::reference_internal)
        .def("getProportions", &Grid::getProportions, py::return_value_policy::reference_internal)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Grid >)
        // TODO: if combined, the return type would depend on the runtime parameter (entity)
        .def("getEntity_cell", &Grid::template getEntity<typename Grid::Cell>)
        .def("getEntity_face", &Grid::template getEntity<typename Grid::Face>)
        .def("getEntity_vertex", &Grid::template getEntity<typename Grid::Vertex>)
        .def("getEntityIndex", &Grid::template getEntityIndex<typename Grid::Cell>)
        .def("getEntityIndex", &Grid::template getEntityIndex<typename Grid::Face>)
        .def("getEntityIndex", &Grid::template getEntityIndex<typename Grid::Vertex>)
        .def("getCellMeasure", &Grid::getCellMeasure, py::return_value_policy::reference_internal)
        .def("getSpaceSteps", &Grid::getSpaceSteps, py::return_value_policy::reference_internal)
        .def("getSmallestSpaceStep", &Grid::getSmallestSpaceStep)
        // TODO: writeProlog()
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

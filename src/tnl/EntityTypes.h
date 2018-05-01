#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

enum class EntityTypes { Cell, Face, Vertex };

inline void
export_EntityTypes( py::module & m )
{
    // avoid duplicate conversion -> export only once
    static bool exported = false;
    if( ! exported ) {
        // TODO: should be nested types instead
        py::enum_< EntityTypes >( m, "EntityTypes" )
            .value("Cell", EntityTypes::Cell)
            .value("Face", EntityTypes::Face)
            .value("Vertex", EntityTypes::Vertex)
        ;
        exported = true;
    }
}

template< typename Mesh >
typename Mesh::GlobalIndexType
mesh_getEntitiesCount( const Mesh & self, const EntityTypes & entity )
{
    if( entity == EntityTypes::Cell )
        return self.template getEntitiesCount< typename Mesh::Cell >();
    else if( entity == EntityTypes::Face )
        return self.template getEntitiesCount< typename Mesh::Face >();
    else if( entity == EntityTypes::Vertex )
        return self.template getEntitiesCount< typename Mesh::Vertex >();
    else
        throw py::value_error("The entity parameter must be either Cell, Face or Vertex.");
}

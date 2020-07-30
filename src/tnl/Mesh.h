#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "../typedefs.h"
#include "StaticVector.h"
#include "EntityTypes.h"

#include <TNL/String.h>
#include <TNL/Meshes/Geometry/getEntityCenter.h>
#include <TNL/Meshes/Geometry/getEntityMeasure.h>

#include <type_traits>

template< typename MeshEntity,
          int Superdimension,
          typename Scope,
          std::enable_if_t< Superdimension <= MeshEntity::MeshType::getMeshDimension(), bool > = true >
//                                           && MeshEntity::template SuperentityTraits< Superdimension >::storageEnabled >::type >
void export_getSuperentityIndex( Scope & m )
{
    m.def("getSuperentityIndex", []( const MeshEntity& entity, const typename MeshEntity::LocalIndexType& i ) {
                                        return entity.template getSuperentityIndex< Superdimension >( i );
                                    }
            );
}

template< typename MeshEntity,
          int Superdimension,
          typename Scope,
          std::enable_if_t< ! ( Superdimension <= MeshEntity::MeshType::getMeshDimension() ), bool > = true >
void export_getSuperentityIndex( Scope & )
{
}


template< typename MeshEntity,
          int Subdimension,
          typename Scope,
          std::enable_if_t< Subdimension <= MeshEntity::MeshType::getMeshDimension()
                            && (Subdimension < MeshEntity::getEntityDimension()), bool > = true >
void export_getSubentityIndex( Scope & m, const char* name )
{
    m.def(name, []( const MeshEntity& entity, const typename MeshEntity::LocalIndexType& i ) {
                    return entity.template getSubentityIndex< Subdimension >( i );
                }
            );
}

template< typename MeshEntity,
          int Subdimension,
          typename Scope,
          std::enable_if_t< ! ( Subdimension <= MeshEntity::MeshType::getMeshDimension()
                                && (Subdimension < MeshEntity::getEntityDimension())
                              ), bool > = true >
void export_getSubentityIndex( Scope &, const char* )
{
}


template< typename MeshEntity,
          typename Scope,
          std::enable_if_t< MeshEntity::getEntityDimension() == 0, bool > = true >
void export_getPoint( Scope & scope )
{
    scope.def("getPoint", []( const MeshEntity& entity ) {
                            return entity.getPoint();
                        }
            );
}

template< typename MeshEntity,
          typename Scope,
          std::enable_if_t< MeshEntity::getEntityDimension() != 0, bool > = true >
void export_getPoint( Scope & )
{
}


template< typename MeshEntity, typename Scope >
void export_MeshEntity( Scope & scope, const char* name )
{
    auto entity = py::class_< MeshEntity >( scope, name )
        .def_static("getEntityDimension", &MeshEntity::getEntityDimension)
        .def("getIndex", &MeshEntity::getIndex)
        // TODO
    ;

    export_getSuperentityIndex< MeshEntity, MeshEntity::getEntityDimension() + 1 >( entity );
    export_getSubentityIndex< MeshEntity, 0 >( entity, "getSubvertexIndex" );
    export_getPoint< MeshEntity >( entity );
}

template< typename Mesh >
void export_Mesh( py::module & m, const char* name )
{
    // there are two templates - const and non-const - take only the const
    auto (Mesh::* getEntity_cell)(const typename Mesh::GlobalIndexType) const = &Mesh::template getEntity<typename Mesh::Cell>;
    auto (Mesh::* getEntity_face)(const typename Mesh::GlobalIndexType) const = &Mesh::template getEntity<typename Mesh::Face>;
    auto (Mesh::* getEntity_vertex)(const typename Mesh::GlobalIndexType) const = &Mesh::template getEntity<typename Mesh::Vertex>;

    export_EntityTypes(m);

    auto mesh = py::class_< Mesh, TNL::Object >( m, name )
        .def(py::init<>())
        .def_static("getMeshDimension", &Mesh::getMeshDimension)
        .def_static("getSerializationType", &Mesh::getSerializationType)
        .def("getSerializationTypeVirtual", &Mesh::getSerializationTypeVirtual)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Mesh >)
        // TODO: if combined, the return type would depend on the runtime parameter (entity)
        .def("getEntity_cell", getEntity_cell)
        .def("getEntity_face", getEntity_face)
        .def("getEntity_vertex", getEntity_vertex)
        .def("getEntityCenter", []( const Mesh& mesh, const typename Mesh::Cell& cell ){ return getEntityCenter( mesh, cell ); } )
        .def("getEntityCenter", []( const Mesh& mesh, const typename Mesh::Face& face ){ return getEntityCenter( mesh, face ); } )
        .def("getEntityCenter", []( const Mesh& mesh, const typename Mesh::Vertex& vertex ){ return getEntityCenter( mesh, vertex ); } )
        .def("getEntityMeasure", []( const Mesh& mesh, const typename Mesh::Cell& cell ){ return getEntityMeasure( mesh, cell ); } )
        .def("getEntityMeasure", []( const Mesh& mesh, const typename Mesh::Face& face ){ return getEntityMeasure( mesh, face ); } )
        .def("getEntityMeasure", []( const Mesh& mesh, const typename Mesh::Vertex& vertex ){ return getEntityMeasure( mesh, vertex ); } )
        .def("isBoundaryEntity", []( const Mesh& mesh, const typename Mesh::Cell& cell ){
                                       return mesh.template isBoundaryEntity< Mesh::Cell::getEntityDimension() >( cell.getIndex() ); } )
        .def("isBoundaryEntity", []( const Mesh& mesh, const typename Mesh::Face& face ){
                                       return mesh.template isBoundaryEntity< Mesh::Face::getEntityDimension() >( face.getIndex() ); } )
        .def("isBoundaryEntity", []( const Mesh& mesh, const typename Mesh::Vertex& vertex ){
                                        return mesh.template isBoundaryEntity< Mesh::Vertex::getEntityDimension() >( vertex.getIndex() ); } )
        // TODO: more?
    ;

    // nested types
    export_MeshEntity< typename Mesh::Cell >( mesh, "Cell" );
    export_MeshEntity< typename Mesh::Face >( mesh, "Face" );
    // avoid duplicate conversion if the type is the same
    if( ! std::is_same< typename Mesh::Face, typename Mesh::Vertex >::value )
        export_MeshEntity< typename Mesh::Vertex >( mesh, "Vertex" );
}

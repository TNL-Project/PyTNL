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

template< typename MeshEntity >
typename MeshEntity::MeshTraitsType::GlobalIndexType
getIndex( const MeshEntity& entity )
{
    return entity.getIndex();
};

struct _general {};
struct _special : _general {};

template< typename MeshEntity,
          int Superdimension,
          typename Scope,
          typename = typename std::enable_if< Superdimension <= MeshEntity::MeshTraitsType::meshDimension >::type >
//                                           && MeshEntity::template SuperentityTraits< Superdimension >::storageEnabled >::type >
void export_getSuperentityIndex( Scope & m, _special )
{
    m.def("getSuperentityIndex", []( const MeshEntity& entity, const typename MeshEntity::LocalIndexType& i ) {
                                        return entity.template getSuperentityIndex< Superdimension >( i );
                                    }
            );
}

template< typename MeshEntity,
          int Superdimension,
          typename Scope >
void export_getSuperentityIndex( Scope &, _general )
{
}


template< typename MeshEntity,
          int Subdimension,
          typename Scope,
          typename = typename std::enable_if< Subdimension <= MeshEntity::MeshTraitsType::meshDimension
                                              && (Subdimension < MeshEntity::getEntityDimension()) >::type >
void export_getSubentityIndex( Scope & m, const char* name, _special )
{
    m.def(name, []( const MeshEntity& entity, const typename MeshEntity::LocalIndexType& i ) {
                    return entity.template getSubentityIndex< Subdimension >( i );
                }
            );
}

template< typename MeshEntity,
          int Superdimension,
          typename Scope >
void export_getSubentityIndex( Scope &, const char*, _general )
{
}


template< typename MeshEntity,
          typename Scope,
          typename = typename std::enable_if< MeshEntity::getEntityDimension() == 0 >::type >
void export_getPoint( Scope & scope, _special )
{
    scope.def("getPoint", []( const MeshEntity& entity ) {
                            return entity.getPoint();
                        }
            );
}

template< typename MeshEntity,
          typename Scope >
void export_getPoint( Scope &, _general )
{
}


template< typename MeshEntity, typename Scope >
void export_MeshEntity( Scope & scope, const char* name )
{
    auto entity = py::class_< MeshEntity >( scope, name )
        .def_static("getEntityDimension", &MeshEntity::getEntityDimension)
        // FIXME: boost chokes on this
//        .def("getIndex", &MeshEntity::getIndex, py::return_internal_reference<>())
        .def("getIndex", getIndex< MeshEntity >)
        // TODO
    ;

    export_getSuperentityIndex< MeshEntity, MeshEntity::getEntityDimension() + 1 >( entity, _special() );
    export_getSubentityIndex< MeshEntity, 0 >( entity, "getSubvertexIndex", _special() );
    export_getPoint< MeshEntity >( entity, _special() );
}

template< typename Mesh >
void export_Mesh( py::module & m, const char* name )
{
    // there are two templates - const and non-const - take only the const
    auto (Mesh::* getEntity_cell)(const typename Mesh::GlobalIndexType&) const = &Mesh::template getEntity<typename Mesh::Cell>;
    auto (Mesh::* getEntity_face)(const typename Mesh::GlobalIndexType&) const = &Mesh::template getEntity<typename Mesh::Face>;
    auto (Mesh::* getEntity_vertex)(const typename Mesh::GlobalIndexType&) const = &Mesh::template getEntity<typename Mesh::Vertex>;

    export_EntityTypes(m);

    auto mesh = py::class_< Mesh, TNL::Object >( m, name )
        .def(py::init<>())
        .def_static("getMeshDimension", &Mesh::getMeshDimension)
        .def_static("getSerializationType", &Mesh::getSerializationType)
        .def("getSerializationTypeVirtual", &Mesh::getSerializationTypeVirtual)
        .def("getEntitiesCount", &mesh_getEntitiesCount< Mesh >)
        // TODO: if combined, the return type would depend on the runtime parameter (entity)
        .def("getEntity_cell", getEntity_cell, py::return_value_policy::reference_internal)
        .def("getEntity_face", getEntity_face, py::return_value_policy::reference_internal)
        .def("getEntity_vertex", getEntity_vertex, py::return_value_policy::reference_internal)
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

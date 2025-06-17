#pragma once

#include <pytnl/pytnl.h>

template< typename Mesh >
void
export_getEntitiesCount( nb::class_< Mesh >& scope )
{
   scope  //
      .def( "getEntitiesCount",
            []( Mesh& self, nb::type_object_t< typename Mesh::Cell > entity_type )
            {
               return self.template getEntitiesCount< Mesh::Cell::getEntityDimension() >();
            } )
      .def( "getEntitiesCount",
            []( Mesh& self, nb::type_object_t< typename Mesh::Face > entity_type )
            {
               return self.template getEntitiesCount< Mesh::Face::getEntityDimension() >();
            } )
      .def( "getEntitiesCount",
            []( Mesh& self, nb::type_object_t< typename Mesh::Vertex > entity_type )
            {
               return self.template getEntitiesCount< Mesh::Vertex::getEntityDimension() >();
            } );
}

template< typename Mesh >
void
export_getGhostEntitiesCount( nb::class_< Mesh >& scope )
{
   scope  //
      .def( "getGhostEntitiesCount",
            []( Mesh& self, nb::type_object_t< typename Mesh::Cell > entity_type )
            {
               return self.template getGhostEntitiesCount< Mesh::Cell::getEntityDimension() >();
            } )
      .def( "getGhostEntitiesCount",
            []( Mesh& self, nb::type_object_t< typename Mesh::Face > entity_type )
            {
               return self.template getGhostEntitiesCount< Mesh::Face::getEntityDimension() >();
            } )
      .def( "getGhostEntitiesCount",
            []( Mesh& self, nb::type_object_t< typename Mesh::Vertex > entity_type )
            {
               return self.template getGhostEntitiesCount< Mesh::Vertex::getEntityDimension() >();
            } );
}

template< typename Mesh >
void
export_getGhostEntitiesOffset( nb::class_< Mesh >& scope )
{
   scope  //
      .def( "getGhostEntitiesOffset",
            []( Mesh& self, nb::type_object_t< typename Mesh::Cell > entity_type )
            {
               return self.template getGhostEntitiesOffset< Mesh::Cell::getEntityDimension() >();
            } )
      .def( "getGhostEntitiesOffset",
            []( Mesh& self, nb::type_object_t< typename Mesh::Face > entity_type )
            {
               return self.template getGhostEntitiesOffset< Mesh::Face::getEntityDimension() >();
            } )
      .def( "getGhostEntitiesOffset",
            []( Mesh& self, nb::type_object_t< typename Mesh::Vertex > entity_type )
            {
               return self.template getGhostEntitiesOffset< Mesh::Vertex::getEntityDimension() >();
            } );
}

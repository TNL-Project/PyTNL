// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "Mesh.h"
#include <TNL/Meshes/Readers/VTKReader.h>
#include <TNL/Meshes/Readers/VTUReader.h>

template< typename Reader >
void export_reader( py::module & m, const char* name )
{
    py::class_< Reader >( m, name )
        .def(py::init<std::string>())
        .def("loadMesh", &Reader::template loadMesh< MeshOfEdges >)
        .def("loadMesh", &Reader::template loadMesh< MeshOfTriangles >)
        .def("loadMesh", &Reader::template loadMesh< MeshOfTetrahedrons >)
//        .def("loadMesh", []( Reader& reader, const std::string& name, MeshOfEdges & mesh ) {
//                return reader.loadMesh( name.c_str(), mesh );
//            } )
//        .def("loadMesh", []( Reader& reader, const std::string& name, MeshOfTriangles & mesh ) {
//                return reader.loadMesh( name.c_str(), mesh );
//            } )
//        .def("loadMesh", []( Reader& reader, const std::string& name, MeshOfTetrahedrons & mesh ) {
//                return reader.loadMesh( name.c_str(), mesh );
//            } )
    ;
}

void export_Meshes( py::module & m )
{
    export_Mesh< MeshOfEdges >( m, "MeshOfEdges" );
    export_Mesh< MeshOfTriangles >( m, "MeshOfTriangles" );
    export_Mesh< MeshOfTetrahedrons >( m, "MeshOfTetrahedrons" );

    export_reader< TNL::Meshes::Readers::VTKReader >( m, "VTKReader" );
    export_reader< TNL::Meshes::Readers::VTUReader >( m, "VTUReader" );
}

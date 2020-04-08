// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "Mesh.h"
#include <TNL/Meshes/Readers/VTKReader.h>

void export_Meshes( py::module & m )
{
    export_Mesh< MeshOfEdges >( m, "MeshOfEdges" );
    export_Mesh< MeshOfTriangles >( m, "MeshOfTriangles" );
    export_Mesh< MeshOfTetrahedrons >( m, "MeshOfTetrahedrons" );

    using Reader = TNL::Meshes::Readers::VTKReader;

    py::class_< Reader >( m, "VTKReader" )
        .def(py::init<std::string>())
        .def("readMesh", &Reader::template readMesh< MeshOfEdges >)
        .def("readMesh", &Reader::template readMesh< MeshOfTriangles >)
        .def("readMesh", &Reader::template readMesh< MeshOfTetrahedrons >)
//        .def("readMesh", []( Reader& reader, const std::string& name, MeshOfEdges & mesh ) {
//                return reader.readMesh( name.c_str(), mesh );
//            } )
//        .def("readMesh", []( Reader& reader, const std::string& name, MeshOfTriangles & mesh ) {
//                return reader.readMesh( name.c_str(), mesh );
//            } )
//        .def("readMesh", []( Reader& reader, const std::string& name, MeshOfTetrahedrons & mesh ) {
//                return reader.readMesh( name.c_str(), mesh );
//            } )
    ;
}

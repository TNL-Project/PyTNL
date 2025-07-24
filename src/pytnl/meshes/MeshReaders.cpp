#include <pytnl/pytnl.h>
#include <pytnl/meshes/MeshReaders.h>

#include <TNL/Meshes/Readers/getMeshReader.h>

void
export_MeshReaders( nb::module_& m )
{
   using MeshReader = TNL::Meshes::Readers::MeshReader;
   using XMLVTK = TNL::Meshes::Readers::XMLVTK;

   // base class with trampolines for virtual methods
   nb::class_< MeshReader, PyMeshReader >( m, "MeshReader" )
      .def( nb::init< std::string >() )
      // bindings against the actual class, NOT the trampoline
      .def( "reset", &MeshReader::reset )
      .def( "detectMesh", &MeshReader::detectMesh )
      .def( "loadMesh", &MeshReader::template loadMesh< Grid_1_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< Grid_2_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< Grid_3_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfEdges_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfTriangles_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfQuadrangles_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfTetrahedrons_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfHexahedrons_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfPolygons_host > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfPolyhedrons_host > )
      .def( "readPointData", &MeshReader::readPointData )
      .def( "readCellData", &MeshReader::readCellData );

   nb::class_< TNL::Meshes::Readers::VTKReader, MeshReader >( m, "VTKReader" ).def( nb::init< std::string >() );

   // base class for VTUReader, VTIReader and PVTUReader
   nb::class_< XMLVTK, PyXMLVTK, MeshReader >( m, "XMLVTK" ).def( nb::init< std::string >() );

   nb::class_< TNL::Meshes::Readers::VTUReader, XMLVTK >( m, "VTUReader" ).def( nb::init< std::string >() );

   nb::class_< TNL::Meshes::Readers::VTIReader, XMLVTK >( m, "VTIReader" ).def( nb::init< std::string >() );

   auto getMeshReader =  //
      m.def( "getMeshReader",
             TNL::Meshes::Readers::getMeshReader,
             nb::arg( "file_name" ),
             nb::kw_only(),
             nb::arg( "file_format" ) = "auto",
             "Returns the MeshReader instance for given file based on file extension "
             "(does not call `reader.detectMesh` so it succeeds even for invalid file)" );
}

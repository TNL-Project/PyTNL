// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include <pytnl/tnl/MeshReaders.h>
#include <pytnl/nanobind.h>
#include <pytnl/typedefs.h>

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
      .def( "loadMesh", &MeshReader::template loadMesh< Grid1D > )
      .def( "loadMesh", &MeshReader::template loadMesh< Grid2D > )
      .def( "loadMesh", &MeshReader::template loadMesh< Grid3D > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfEdges > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfTriangles > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfQuadrangles > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfTetrahedrons > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfHexahedrons > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfPolygons > )
      .def( "loadMesh", &MeshReader::template loadMesh< MeshOfPolyhedrons > )
      .def( "readPointData", &MeshReader::readPointData )
      .def( "readCellData", &MeshReader::readCellData );

   nb::class_< TNL::Meshes::Readers::VTKReader, MeshReader >( m, "VTKReader" ).def( nb::init< std::string >() );

   // base class for VTUReader, VTIReader and PVTUReader
   nb::class_< XMLVTK, PyXMLVTK, MeshReader >( m, "XMLVTK" ).def( nb::init< std::string >() );

   nb::class_< TNL::Meshes::Readers::VTUReader, XMLVTK >( m, "VTUReader" ).def( nb::init< std::string >() );

   nb::class_< TNL::Meshes::Readers::VTIReader, XMLVTK >( m, "VTIReader" ).def( nb::init< std::string >() );
}

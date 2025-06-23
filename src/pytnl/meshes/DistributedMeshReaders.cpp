#include <pytnl/pytnl.h>
#include <pytnl/meshes/MeshReaders.h>

#include <TNL/Meshes/Readers/PVTUReader.h>

void
export_DistributedMeshReaders( nb::module_& m )
{
   using XMLVTK = TNL::Meshes::Readers::XMLVTK;
   using PVTUReader = TNL::Meshes::Readers::PVTUReader;

   nb::class_< PVTUReader, XMLVTK >( m, "PVTUReader" )
      .def( nb::init< std::string >() )
      // loadMesh is not virtual in PVTUReader
      .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfEdges > )
      .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTriangles > )
      .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfQuadrangles > )
      .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTetrahedrons > )
      .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfHexahedrons > );
}

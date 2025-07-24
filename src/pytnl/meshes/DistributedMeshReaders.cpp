#include <pytnl/pytnl.h>
#include <pytnl/meshes/MeshReaders.h>

#include <TNL/Meshes/Readers/PVTUReader.h>

template< typename Mesh, typename Reader, typename... Args >
void
export_dummy_loadMesh( nb::class_< Reader, Args... >& reader )
{
   reader.def( "loadMesh",
               []( const Reader& self, Mesh& mesh )
               {
                  throw std::logic_error( "cannot load non-distributed mesh using a distributed mesh reader" );
               } );
}

void
export_DistributedMeshReaders( nb::module_& m )
{
   using XMLVTK = TNL::Meshes::Readers::XMLVTK;
   using PVTUReader = TNL::Meshes::Readers::PVTUReader;

   auto reader =  //
      nb::class_< PVTUReader, XMLVTK >( m, "PVTUReader" )
         .def( nb::init< std::string >() )
         // loadMesh is not virtual in PVTUReader
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfEdges_host > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTriangles_host > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfQuadrangles_host > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTetrahedrons_host > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfHexahedrons_host > );

   // Add overloads for all types that loadMesh in the base class can handle to make mypy happy
   export_dummy_loadMesh< Grid_1_host >( reader );
   export_dummy_loadMesh< Grid_2_host >( reader );
   export_dummy_loadMesh< Grid_3_host >( reader );
   export_dummy_loadMesh< MeshOfEdges_host >( reader );
   export_dummy_loadMesh< MeshOfTriangles_host >( reader );
   export_dummy_loadMesh< MeshOfQuadrangles_host >( reader );
   export_dummy_loadMesh< MeshOfTetrahedrons_host >( reader );
   export_dummy_loadMesh< MeshOfHexahedrons_host >( reader );
   export_dummy_loadMesh< MeshOfPolygons_host >( reader );
   export_dummy_loadMesh< MeshOfPolyhedrons_host >( reader );
}

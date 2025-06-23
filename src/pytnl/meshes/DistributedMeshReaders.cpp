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
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfEdges > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTriangles > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfQuadrangles > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfTetrahedrons > )
         .def( "loadMesh", &PVTUReader::template loadMesh< DistributedMeshOfHexahedrons > );

   // Add overloads for all types that loadMesh in the base class can handle to make mypy happy,
   export_dummy_loadMesh< Grid1D >( reader );
   export_dummy_loadMesh< Grid2D >( reader );
   export_dummy_loadMesh< Grid3D >( reader );
   export_dummy_loadMesh< MeshOfEdges >( reader );
   export_dummy_loadMesh< MeshOfTriangles >( reader );
   export_dummy_loadMesh< MeshOfQuadrangles >( reader );
   export_dummy_loadMesh< MeshOfTetrahedrons >( reader );
   export_dummy_loadMesh< MeshOfHexahedrons >( reader );
   export_dummy_loadMesh< MeshOfPolygons >( reader );
   export_dummy_loadMesh< MeshOfPolyhedrons >( reader );
}

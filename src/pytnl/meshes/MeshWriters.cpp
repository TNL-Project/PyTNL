#include "MeshWriters.h"

void
export_MeshWriters( nb::module_& m )
{
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid_1_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_Grid_1" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid_1_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_Grid_1" );
   export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid_1_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTIWriter_Grid_1" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid_2_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_Grid_2" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid_2_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_Grid_2" );
   export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid_2_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTIWriter_Grid_2" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid_3_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_Grid_3" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid_3_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_Grid_3" );
   export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid_3_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTIWriter_Grid_3" );

   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfEdges_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_MeshOfEdges" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfEdges_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_MeshOfEdges" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfTriangles_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_MeshOfTriangles" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfTriangles_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_MeshOfTriangles" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfQuadrangles_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_MeshOfQuadrangles" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfQuadrangles_host >,
                      TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfQuadrangles" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfTetrahedrons_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_MeshOfTetrahedrons" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfTetrahedrons_host >,
                      TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfTetrahedrons" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfHexahedrons_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_MeshOfHexahedrons" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfHexahedrons_host >,
                      TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfHexahedrons" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfPolygons_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_MeshOfPolygons" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfPolygons_host >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_MeshOfPolygons" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< MeshOfPolyhedrons_host >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_MeshOfPolyhedrons" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< MeshOfPolyhedrons_host >,
                      TNL::Meshes::VTK::FileFormat::zlib_compressed >( m, "VTUWriter_MeshOfPolyhedrons" );
}

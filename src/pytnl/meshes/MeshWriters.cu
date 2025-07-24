#include "MeshWriters.h"

void
export_MeshWriters( nb::module_& m )
{
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid_1_cuda >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_Grid_1" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid_1_cuda >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_Grid_1" );
   export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid_1_cuda >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTIWriter_Grid_1" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid_2_cuda >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_Grid_2" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid_2_cuda >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_Grid_2" );
   export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid_2_cuda >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTIWriter_Grid_2" );
   export_MeshWriter< TNL::Meshes::Writers::VTKWriter< Grid_3_cuda >, TNL::Meshes::VTK::FileFormat::binary >(
      m, "VTKWriter_Grid_3" );
   export_MeshWriter< TNL::Meshes::Writers::VTUWriter< Grid_3_cuda >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTUWriter_Grid_3" );
   export_MeshWriter< TNL::Meshes::Writers::VTIWriter< Grid_3_cuda >, TNL::Meshes::VTK::FileFormat::zlib_compressed >(
      m, "VTIWriter_Grid_3" );

   // Writers for unstructured meshes do not work with CUDA
}

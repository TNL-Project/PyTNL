#include "DistributedMeshWriters.h"

void
export_DistributedMeshWriters( nb::module_& m )
{
   constexpr TNL::Meshes::VTK::FileFormat default_format = TNL::Meshes::VTK::FileFormat::zlib_compressed;
   export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfEdges_cuda, default_format >(
      m, "PVTUWriter_MeshOfEdges" );
   export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfTriangles_cuda, default_format >(
      m, "PVTUWriter_MeshOfTriangles" );
   export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfQuadrangles_cuda, default_format >(
      m, "PVTUWriter_MeshOfQuadrangles" );
   export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfTetrahedrons_cuda, default_format >(
      m, "PVTUWriter_MeshOfTetrahedrons" );
   export_DistributedMeshWriter< TNL::Meshes::Writers::PVTUWriter, MeshOfHexahedrons_cuda, default_format >(
      m, "PVTUWriter_MeshOfHexahedrons" );
}

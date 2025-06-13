#include <pytnl/exceptions.h>
#include <pytnl/typedefs.h>

// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

// external functions
void
export_Grid1D( nb::module_& m );
void
export_Grid2D( nb::module_& m );
void
export_Grid3D( nb::module_& m );
void
export_VTKTraits( nb::module_& m );
void
export_Meshes( nb::module_& m );
void
export_MeshReaders( nb::module_& m );
void
export_MeshWriters( nb::module_& m );

// Python module definition
NB_MODULE( meshes, m )
{
   register_exceptions( m );

   export_Grid1D( m );
   export_Grid2D( m );
   export_Grid3D( m );

   export_VTKTraits( m );

   export_Meshes( m );
   export_MeshReaders( m );
   export_MeshWriters( m );
}

// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include <pytnl/tnl/Grid.h>

void
export_Grid3D( py::module& m )
{
   export_Grid< Grid3D >( m, "Grid3D" );
}

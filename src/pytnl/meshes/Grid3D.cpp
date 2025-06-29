#include "Grid.h"

void
export_Grid3D( nb::module_& m )
{
   export_Grid< Grid3D >( m, "Grid_3" );
}

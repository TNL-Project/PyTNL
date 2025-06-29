#include "Grid.h"

void
export_Grid2D( nb::module_& m )
{
   export_Grid< Grid2D >( m, "Grid_2" );
}

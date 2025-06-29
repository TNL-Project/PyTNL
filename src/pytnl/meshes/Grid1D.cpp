#include "Grid.h"

void
export_Grid1D( nb::module_& m )
{
   export_Grid< Grid1D >( m, "Grid_1" );
}

// conversions have to be registered for each object file
#include <pytnl/tnl_conversions.h>

#include "Grid.h"

void
export_Grid1D( nb::module_& m )
{
   export_Grid< Grid1D >( m, "Grid1D" );
}

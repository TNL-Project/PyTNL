#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include "IterativeSolver.h"
#include "ExplicitSolver.h"
#include "ODESolver.h"

#include <TNL/Solvers/ODE/Methods/Euler.h>

using Vector = TNL::Containers::Vector< RealType, TNL::Devices::Host, IndexType >;
using VectorView = typename Vector::ViewType;
using Euler = TNL::Solvers::ODE::Methods::Euler< RealType >;

// Python module definition
NB_MODULE( _solvers, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers" );

   export_IterativeSolver< RealType, IndexType >( m, "IterativeSolver_float_int" );

   export_ExplicitSolver< RealType, IndexType >( m, "ExplicitSolver_float_int" );

   export_ODESolver< Euler, Vector >( m, "ODESolver_Euler" );
}

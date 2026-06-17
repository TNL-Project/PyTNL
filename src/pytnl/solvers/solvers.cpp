#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>

#include "IterativeSolver.h"
#include "ExplicitSolver.h"
#include "ODESolver.h"

#include <TNL/Solvers/ODE/Methods/BogackiShampin.h>
#include <TNL/Solvers/ODE/Methods/CashKarp.h>
#include <TNL/Solvers/ODE/Methods/DormandPrince.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>
#include <TNL/Solvers/ODE/Methods/Fehlberg2.h>
#include <TNL/Solvers/ODE/Methods/Fehlberg5.h>
#include <TNL/Solvers/ODE/Methods/Heun2.h>
#include <TNL/Solvers/ODE/Methods/Heun3.h>
#include <TNL/Solvers/ODE/Methods/Kutta.h>
#include <TNL/Solvers/ODE/Methods/KuttaMerson.h>
#include <TNL/Solvers/ODE/Methods/Midpoint.h>
#include <TNL/Solvers/ODE/Methods/OriginalRungeKutta.h>
#include <TNL/Solvers/ODE/Methods/Ralston2.h>
#include <TNL/Solvers/ODE/Methods/Ralston3.h>
#include <TNL/Solvers/ODE/Methods/Ralston4.h>
#include <TNL/Solvers/ODE/Methods/Rule38.h>
#include <TNL/Solvers/ODE/Methods/SSPRK3.h>
#include <TNL/Solvers/ODE/Methods/VanDerHouwenWray.h>

using Vector = TNL::Containers::Vector< RealType, TNL::Devices::Host, IndexType >;
using VectorView = typename Vector::ViewType;
using BogackiShampin = TNL::Solvers::ODE::Methods::BogackiShampin< RealType >;
using CashKarp = TNL::Solvers::ODE::Methods::CashKarp< RealType >;
using DormandPrince = TNL::Solvers::ODE::Methods::DormandPrince< RealType >;
using Euler = TNL::Solvers::ODE::Methods::Euler< RealType >;
using Fehlberg2 = TNL::Solvers::ODE::Methods::Fehlberg2< RealType >;
using Fehlberg5 = TNL::Solvers::ODE::Methods::Fehlberg5< RealType >;
using Heun2 = TNL::Solvers::ODE::Methods::Heun2< RealType >;
using Heun3 = TNL::Solvers::ODE::Methods::Heun3< RealType >;
using Kutta = TNL::Solvers::ODE::Methods::Kutta< RealType >;
using KuttaMerson = TNL::Solvers::ODE::Methods::KuttaMerson< RealType >;
using Midpoint = TNL::Solvers::ODE::Methods::Midpoint< RealType >;
using OriginalRungeKutta = TNL::Solvers::ODE::Methods::OriginalRungeKutta< RealType >;
using Ralston2 = TNL::Solvers::ODE::Methods::Ralston2< RealType >;
using Ralston3 = TNL::Solvers::ODE::Methods::Ralston3< RealType >;
using Ralston4 = TNL::Solvers::ODE::Methods::Ralston4< RealType >;
using Rule38 = TNL::Solvers::ODE::Methods::Rule38< RealType >;
using SSPRK3 = TNL::Solvers::ODE::Methods::SSPRK3< RealType >;
using VanDerHouwenWray = TNL::Solvers::ODE::Methods::VanDerHouwenWray< RealType >;

// Python module definition
NB_MODULE( _solvers, m )
{
   register_exceptions( m );

   // import depending modules
   nb::module_::import_( "pytnl._containers" );

   export_IterativeSolver< RealType, IndexType >( m, "IterativeSolver_float_int" );

   export_ExplicitSolver< RealType, IndexType >( m, "ExplicitSolver_float_int" );

   export_ODESolver< BogackiShampin, Vector >( m, "ODESolver_BogackiShampin" );
   export_ODESolver< CashKarp, Vector >( m, "ODESolver_CashKarp" );
   export_ODESolver< DormandPrince, Vector >( m, "ODESolver_DormandPrince" );
   export_ODESolver< Euler, Vector >( m, "ODESolver_Euler" );
   export_ODESolver< Fehlberg2, Vector >( m, "ODESolver_Fehlberg2" );
   export_ODESolver< Fehlberg5, Vector >( m, "ODESolver_Fehlberg5" );
   export_ODESolver< Heun2, Vector >( m, "ODESolver_Heun2" );
   export_ODESolver< Heun3, Vector >( m, "ODESolver_Heun3" );
   export_ODESolver< Kutta, Vector >( m, "ODESolver_Kutta" );
   export_ODESolver< KuttaMerson, Vector >( m, "ODESolver_KuttaMerson" );
   export_ODESolver< Midpoint, Vector >( m, "ODESolver_Midpoint" );
   export_ODESolver< OriginalRungeKutta, Vector >( m, "ODESolver_OriginalRungeKutta" );
   export_ODESolver< Ralston2, Vector >( m, "ODESolver_Ralston2" );
   export_ODESolver< Ralston3, Vector >( m, "ODESolver_Ralston3" );
   export_ODESolver< Ralston4, Vector >( m, "ODESolver_Ralston4" );
   export_ODESolver< Rule38, Vector >( m, "ODESolver_Rule38" );
   export_ODESolver< SSPRK3, Vector >( m, "ODESolver_SSPRK3" );
   export_ODESolver< VanDerHouwenWray, Vector >( m, "ODESolver_VanDerHouwenWray" );
}

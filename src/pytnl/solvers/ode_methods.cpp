#include <pytnl/pytnl.h>

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

template< typename Method, typename Scope >
void
export_ode_method( Scope& m, const char* name, const char* doc )
{
   nb::class_< Method >( m, name, doc );
}

void
export_ode_methods( nb::module_& m )
{
   auto submodule = m.def_submodule( "ode_methods" );

   export_ode_method< TNL::Solvers::ODE::Methods::BogackiShampin< RealType > >(
      submodule, "BogackiShampin", "Third-order Bogacki-Shampin method with adaptive step (Matlab ode23)." );
   export_ode_method< TNL::Solvers::ODE::Methods::CashKarp< RealType > >(
      submodule, "CashKarp", "Fifth-order Cash-Karp method with adaptive step." );
   export_ode_method< TNL::Solvers::ODE::Methods::DormandPrince< RealType > >(
      submodule, "DormandPrince", "Fifth-order Dormand-Prince method (Matlab ode45)." );
   export_ode_method< TNL::Solvers::ODE::Methods::Euler< RealType > >( submodule, "Euler", "First-order Euler method." );
   export_ode_method< TNL::Solvers::ODE::Methods::Fehlberg2< RealType > >(
      submodule, "Fehlberg2", "Second-order Fehlberg method with adaptive step." );
   export_ode_method< TNL::Solvers::ODE::Methods::Fehlberg5< RealType > >(
      submodule, "Fehlberg5", "Fifth-order Fehlberg method with adaptive step." );
   export_ode_method< TNL::Solvers::ODE::Methods::Heun2< RealType > >(
      submodule, "Heun2", "Second-order Heun method with adaptive step." );
   export_ode_method< TNL::Solvers::ODE::Methods::Heun3< RealType > >( submodule, "Heun3", "Third-order Heun method." );
   export_ode_method< TNL::Solvers::ODE::Methods::Kutta< RealType > >( submodule, "Kutta", "Third-order Kutta method." );
   export_ode_method< TNL::Solvers::ODE::Methods::KuttaMerson< RealType > >(
      submodule, "KuttaMerson", "Fourth-order Runge-Kutta-Merson method with adaptive step." );
   export_ode_method< TNL::Solvers::ODE::Methods::Midpoint< RealType > >(
      submodule, "Midpoint", "Second-order midpoint method." );
   export_ode_method< TNL::Solvers::ODE::Methods::OriginalRungeKutta< RealType > >(
      submodule, "OriginalRungeKutta", "Classic fourth-order Runge-Kutta method." );
   export_ode_method< TNL::Solvers::ODE::Methods::Ralston2< RealType > >(
      submodule, "Ralston2", "Second-order Ralston method." );
   export_ode_method< TNL::Solvers::ODE::Methods::Ralston3< RealType > >(
      submodule, "Ralston3", "Third-order Ralston method." );
   export_ode_method< TNL::Solvers::ODE::Methods::Ralston4< RealType > >(
      submodule, "Ralston4", "Fourth-order Ralston method." );
   export_ode_method< TNL::Solvers::ODE::Methods::Rule38< RealType > >(
      submodule, "Rule38", "Fourth-order 3/8-rule Runge-Kutta method." );
   export_ode_method< TNL::Solvers::ODE::Methods::SSPRK3< RealType > >(
      submodule, "SSPRK3", "Third-order Strong Stability Preserving Runge-Kutta method." );
   export_ode_method< TNL::Solvers::ODE::Methods::VanDerHouwenWray< RealType > >(
      submodule, "VanDerHouwenWray", "Third-order Van der Houwen-Wray method." );
}

#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Solvers/ODE/ExplicitSolver.h>

template< typename Real, typename Index >
void
export_ExplicitSolver( nb::module_& m, const char* name )
{
   using namespace TNL::Solvers;
   using namespace TNL::Solvers::ODE;

   using SolverMonitor = IterativeSolverMonitor< Real >;
   using ExplicitSolver = ExplicitSolver< Real, Index, SolverMonitor >;
   using IterativeSolver = IterativeSolver< Real, Index, SolverMonitor >;

   auto solver =  //
      nb::class_< ExplicitSolver, IterativeSolver >( m, name )
         .def( nb::init<>() )
         .def( "setTime", &ExplicitSolver::setTime )
         .def( "getTime", &ExplicitSolver::getTime )
         .def( "setStopTime", &ExplicitSolver::setStopTime )
         .def( "getStopTime", &ExplicitSolver::getStopTime )
         .def( "setTau", &ExplicitSolver::setTau )
         .def( "getTau", &ExplicitSolver::getTau )
         .def( "setMaxTau", &ExplicitSolver::setMaxTau )
         .def( "getMaxTau", &ExplicitSolver::getMaxTau )
         .def( "refreshSolverMonitor", &ExplicitSolver::refreshSolverMonitor )
         .def( "checkNextIteration", &ExplicitSolver::checkNextIteration );
}

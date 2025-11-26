#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Solvers/IterativeSolver.h>

template< typename Real, typename Index >
void
export_IterativeSolver( nb::module_& m, const char* name )
{
   using namespace TNL::Solvers;

   using SolverMonitor = IterativeSolverMonitor< Real >;
   using IterativeSolver = IterativeSolver< Real, Index, SolverMonitor >;

   auto solver =  //
      nb::class_< IterativeSolver >( m, name )
         .def( nb::init<>() )
         .def( "setMaxIterations", &IterativeSolver::setMaxIterations )
         .def( "getMaxIterations", &IterativeSolver::getMaxIterations )
         .def( "setMinIterations", &IterativeSolver::setMinIterations )
         .def( "getMinIterations", &IterativeSolver::getMinIterations )
         .def( "getIterations", &IterativeSolver::getIterations )
         .def( "setConvergenceResidue", &IterativeSolver::setConvergenceResidue )
         .def( "getConvergenceResidue", &IterativeSolver::getConvergenceResidue )
         .def( "setDivergenceResidue", &IterativeSolver::setDivergenceResidue )
         .def( "getDivergenceResidue", &IterativeSolver::getDivergenceResidue )
         .def( "setResidue", &IterativeSolver::setResidue )
         .def( "getResidue", &IterativeSolver::getResidue )
         .def( "setRefreshRate", &IterativeSolver::setRefreshRate )
         .def( "setSolverMonitor", &IterativeSolver::setSolverMonitor )
         .def( "resetIterations", &IterativeSolver::resetIterations )
         .def( "nextIteration", &IterativeSolver::nextIteration )
         .def( "checkNextIteration", &IterativeSolver::checkNextIteration )
         .def( "checkConvergence", &IterativeSolver::checkConvergence );
}

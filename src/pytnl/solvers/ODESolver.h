#pragma once

#include <pytnl/pytnl.h>

#include <TNL/Solvers/ODE/ODESolver.h>

template< typename Method, typename Vector >
void
export_ODESolver( nb::module_& m, const char* name )
{
   using namespace TNL::Solvers;
   using namespace TNL::Solvers::ODE;

   using Solver = ODESolver< Method, Vector >;
   using SolverMonitor = IterativeSolverMonitor< typename Vector::RealType >;
   using ExplicitSolver = std::conditional_t<  //
      Solver::isStatic(),
      StaticExplicitSolver< TNL::GetValueType_t< Vector >, std::size_t >,
      ExplicitSolver< typename Vector::RealType, typename Vector::IndexType, SolverMonitor > >;
   using VectorView = typename Vector::ViewType;
   using Real = typename Vector::RealType;

   auto solver =  //
      nb::class_< Solver, ExplicitSolver >( m, name )
         .def( nb::init<>() )
         .def_static( "isStatic", &Solver::isStatic )
         .def( "setAdaptivity", &Solver::setAdaptivity )
         .def( "getAdaptivity", &Solver::getAdaptivity )
         .def( "getMethod", nb::overload_cast<>( &Solver::getMethod, nb::const_ ), nb::rv_policy::reference_internal )
         .def( "init", &Solver::init )
         .def( "reset", &Solver::reset )
         .def( "solve",
               []( Solver& self,
                   Vector& u,
                   std::function< void( const Real& t, const Real& tau, const VectorView& u, VectorView& fu ) > f )
               // TODO: generalize for functions with any number of arguments
               {
                  return self.solve( u, f );
               } )
         .def( "iterate",
               []( Solver& self,
                   Vector& u,
                   std::function< void( const Real& t, const Real& tau, const VectorView& u, VectorView& fu ) > f )
               // TODO: generalize for functions with any number of arguments
               {
                  return self.iterate( u, f );
               } );
}

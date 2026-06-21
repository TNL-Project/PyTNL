// ODE solver benchmark: native C++/CUDA vs equivalent Python benchmark.
//
// Solves the 2D heat equation du/dt = d^2u/dx^2 + d^2u/dy^2 on a 1000x1000
// grid with TNL's ODESolver<Fehlberg2> on both GPU (CUDA) and CPU (Host).
// Fehlberg2 is an adaptive Runge-Kutta method. The RHS is computed via
// TNL::Algorithms::parallelFor, which dispatches to CUDA kernels or OpenMP
// threads depending on the Device template parameter.
//
// Equivalent to examples/ode_solver_cuda.py which uses numba @cuda.jit /
// @jit(nopython=True) for the RHS instead.
//
// Compile (GPU + CPU, requires nvcc):
//   nvcc -std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr --extended-lambda \
//     -arch=sm_120 -Xcompiler -fopenmp -DHAVE_OPENMP \
//     -I build/_deps/tnl-src/src \
//     examples/benchmark_ode_cuda.cu -o examples/benchmark_ode_cuda
//
// Compile (CPU-only, uses g++ with OpenMP):
//   g++ -std=c++17 -O3 -DNDEBUG -fopenmp -DHAVE_OPENMP -x c++ \
//     -I build/_deps/tnl-src/src \
//     examples/benchmark_ode_cuda.cu -o examples/benchmark_ode_cpu

#include <iostream>
#include <chrono>
#include <cmath>

#ifdef __CUDACC__
   #include <cuda_runtime.h>
#endif

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Math.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Fehlberg2.h>

using Real = double;
using Index = int;

template< typename Device >
double
solveHeatEquation( TNL::Containers::Vector< Real, Device, Index >& u_out, Index& iterations )
{
   using Vector = TNL::Containers::Vector< Real, Device, Index >;
   using VectorView = typename Vector::ViewType;
   using Method = TNL::Solvers::ODE::Methods::Fehlberg2< Real >;
   using ODESolver = TNL::Solvers::ODE::ODESolver< Method, Vector >;

   const Index nx = 1000;
   const Index ny = 1000;
   const Index n = nx * ny;
   const Real final_t = 0.001;
   const Real output_time_step = 0.001;
   const Real h = 1.0 / ( nx - 1 );
   const Real tau = 0.25 * h * h;
   const Real h_sqr_inv = 1.0 / ( h * h );
   const Real adaptivity = 0.001;

   // Initial condition: square pulse in [0.4, 0.6] x [0.4, 0.6]
   Vector u( n );
   u.forAllElements(
      [ = ] __cuda_callable__( Index idx, Real & value )
      {
         const Index j = idx / nx;
         const Index i = idx % nx;
         const Real x = i * h;
         const Real y = j * h;
         if( x >= 0.4 && x <= 0.6 && y >= 0.4 && y <= 0.6 )
            value = 1.0;
         else
            value = 0.0;
      } );

   ODESolver solver;
   solver.setTau( tau );
   solver.setTime( 0.0 );
   solver.setAdaptivity( adaptivity );

   auto f = [ = ] __cuda_callable__( Index idx, const VectorView& u, VectorView& fu )
   {
      const Index j = idx / nx;
      const Index i = idx % nx;
      if( i == 0 || i == nx - 1 || j == 0 || j == ny - 1 )
         fu[ idx ] = 0.0;
      else
         fu[ idx ] = h_sqr_inv * ( u[ idx - 1 ] + u[ idx + 1 ] + u[ idx - nx ] + u[ idx + nx ] - 4.0 * u[ idx ] );
   };

   auto time_stepping = [ = ]( const Real& t, const Real& tau, const VectorView& u, VectorView& fu )
   {
      TNL::Algorithms::parallelFor< Device >( 0, n, f, u, fu );
   };

   auto t0 = std::chrono::high_resolution_clock::now();
   while( solver.getTime() < final_t ) {
      solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
      solver.solve( u, time_stepping );
   }
#ifdef __CUDACC__
   if constexpr( std::is_same_v< Device, TNL::Devices::Cuda > )
      cudaDeviceSynchronize();
#endif
   auto t1 = std::chrono::high_resolution_clock::now();

   u_out = std::move( u );
   iterations = solver.getIterations();
   return std::chrono::duration< double >( t1 - t0 ).count();
}

int
main()
{
   const Index n = 1000 * 1000;

#ifdef __CUDACC__
   // --- GPU solve ---
   TNL::Containers::Vector< Real, TNL::Devices::Cuda, Index > u_gpu;
   Index gpu_iterations;
   const double gpu_time = solveHeatEquation< TNL::Devices::Cuda >( u_gpu, gpu_iterations );
#endif

   // --- CPU solve ---
   TNL::Containers::Vector< Real, TNL::Devices::Host, Index > u_cpu;
   Index cpu_iterations;
   const double cpu_time = solveHeatEquation< TNL::Devices::Host >( u_cpu, cpu_iterations );

#ifdef __CUDACC__
   // --- Comparison ---
   TNL::Containers::Vector< Real, TNL::Devices::Host, Index > u_gpu_host( u_gpu );
   Real max_diff = 0.0;
   for( Index i = 0; i < n; i++ )
      max_diff = TNL::max( max_diff, TNL::abs( u_cpu[ i ] - u_gpu_host[ i ] ) );

   std::cout << "Grid: 1000x1000 = " << n << " points, t=0.001" << std::endl;
   std::cout << "GPU iterations: " << gpu_iterations << ", time: " << gpu_time << "s" << std::endl;
   std::cout << "CPU iterations: " << cpu_iterations << ", time: " << cpu_time << "s" << std::endl;
   std::cout << "Speedup:  " << cpu_time / gpu_time << "x" << std::endl;
   std::cout << "Max diff CPU vs GPU: " << max_diff << std::endl;
#else
   std::cout << "Grid: 1000x1000 = " << n << " points, t=0.001" << std::endl;
   std::cout << "CPU iterations: " << cpu_iterations << ", time: " << cpu_time << "s" << std::endl;
   std::cout << "(GPU solve disabled — rebuild with nvcc to enable)" << std::endl;
#endif

   return EXIT_SUCCESS;
}

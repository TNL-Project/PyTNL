"""CUDA ODE solver example: numba @cuda.jit GPU kernel vs numba @jit CPU kernel.

Solves the 2D heat equation du/dt = d^2u/dx^2 + d^2u/dy^2 on a 1000x1000 grid
with ODESolver[Fehlberg2, devices.Cuda] on GPU and ODESolver[Fehlberg2] on
CPU. Fehlberg2 is an adaptive Runge-Kutta method that automatically adjusts
the step size based on local error estimates. Both RHS functions are
JIT-compiled by numba — the GPU kernel uses @cuda.jit with DLPack zero-copy
VectorView, the CPU kernel uses @jit(nopython=True, parallel=True) with prange
for multi-threaded parallelism (equivalent to OpenMP's #pragma omp parallel
for). Timing compares the two approaches.

The CPU kernel uses np.asarray() to create a zero-copy numpy array view of
VectorView's memory via the PEP 3118 buffer protocol. This enables numba to
recognize the data as a numpy ndarray (types.Array), which supports
parallel=True and prange for multi-threaded execution without conversion
overhead.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryComparison=false, reportOperatorIssue=false, reportIndexIssue=false, reportUntypedFunctionDecorator=false, reportUnknownArgumentType=false, reportMissingImports=false
# mypy: disable-error-code="import-untyped, misc, import-not-found"

import time
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
from numba import cuda, jit, prange

from pytnl import devices
from pytnl.containers import Vector
from pytnl.solvers import ODESolver, ode_methods


@jit(nopython=True, parallel=True, fastmath=True)
def _heat2d_rhs_kernel(
    u: np.ndarray,
    fu: np.ndarray,
    nx: int,
    ny: int,
    h_sqr_inv: float,
) -> None:
    for idx in prange(nx * ny):
        j = idx // nx
        i = idx % nx
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            fu[idx] = 0.0
        else:
            fu[idx] = h_sqr_inv * (u[idx - 1] + u[idx + 1] + u[idx - nx] + u[idx + nx] - 4.0 * u[idx])


def heat2d_rhs_cpu(
    t: float,
    tau: float,
    u: Any,  # noqa: ANN401
    fu: Any,  # noqa: ANN401
    nx: int,
    ny: int,
    h_sqr_inv: float,
) -> None:
    _heat2d_rhs_kernel(np.asarray(u), np.asarray(fu), nx, ny, h_sqr_inv)


@cuda.jit
def heat2d_rhs_cuda(
    u: Any,  # noqa: ANN401
    fu: Any,  # noqa: ANN401
    nx: int,
    ny: int,
    h_sqr_inv: float,
) -> None:
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        idx = j * nx + i
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            fu[idx] = 0.0
        else:
            fu[idx] = h_sqr_inv * (u[idx - 1] + u[idx + 1] + u[idx - nx] + u[idx + nx] - 4.0 * u[idx])


def make_cuda_rhs(nx: int, ny: int, h_sqr_inv: float) -> Callable[[float, float, Any, Any], None]:
    threads_per_block = (16, 16)
    blocks_per_grid = (
        (nx + threads_per_block[0] - 1) // threads_per_block[0],
        (ny + threads_per_block[1] - 1) // threads_per_block[1],
    )

    def rhs(t: float, tau: float, u_view: Any, fu_view: Any) -> None:  # noqa: ANN401
        heat2d_rhs_cuda[blocks_per_grid, threads_per_block](u_view, fu_view, nx, ny, h_sqr_inv)

    return rhs


def main() -> None:
    nx = 1000
    ny = 1000
    n = nx * ny
    final_t = 0.001
    output_time_step = 0.001
    h = 1.0 / (nx - 1)
    tau = 0.25 * h * h
    h_sqr_inv = 1.0 / (h * h)
    adaptivity = 0.001

    # Initial condition: square pulse in [0.4, 0.6] x [0.4, 0.6]
    u_cuda = Vector[float, devices.Cuda](n)
    u_cpu = Vector[float](n)
    for j in range(ny):
        y = j * h
        for i in range(nx):
            x = i * h
            val = 1.0 if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6 else 0.0
            u_cuda[j * nx + i] = val
            u_cpu[j * nx + i] = val

    # Warm up JIT compilation for both paths
    warm_u = Vector[float, devices.Cuda](n)
    warm_fu = Vector[float, devices.Cuda](n)
    warm_rhs_cuda = make_cuda_rhs(nx, ny, h_sqr_inv)
    warm_rhs_cuda(0.0, tau, warm_u.getView(), warm_fu.getView())
    cuda.synchronize()

    warm_u_cpu = Vector[float](n)
    warm_fu_cpu = Vector[float](n)
    rhs_cpu = partial(heat2d_rhs_cpu, nx=nx, ny=ny, h_sqr_inv=h_sqr_inv)
    rhs_cpu(0.0, tau, warm_u_cpu.getView(), warm_fu_cpu.getView())

    # --- GPU solve ---
    solver_gpu = ODESolver[ode_methods.Fehlberg2, devices.Cuda]()
    solver_gpu.setTau(tau)
    solver_gpu.setTime(0.0)
    solver_gpu.setAdaptivity(adaptivity)

    t0 = time.perf_counter()
    rhs_gpu = make_cuda_rhs(nx, ny, h_sqr_inv)
    while solver_gpu.getTime() < final_t:
        stop_time = min(solver_gpu.getTime() + output_time_step, final_t)
        solver_gpu.setStopTime(stop_time)
        solver_gpu.solve(u_cuda, rhs_gpu)
    cuda.synchronize()
    gpu_time = time.perf_counter() - t0
    gpu_iterations = solver_gpu.getIterations()

    # --- CPU solve ---
    solver_cpu = ODESolver[ode_methods.Fehlberg2]()
    solver_cpu.setTau(tau)
    solver_cpu.setTime(0.0)
    solver_cpu.setAdaptivity(adaptivity)

    t0 = time.perf_counter()
    while solver_cpu.getTime() < final_t:
        stop_time = min(solver_cpu.getTime() + output_time_step, final_t)
        solver_cpu.setStopTime(stop_time)
        solver_cpu.solve(u_cpu, rhs_cpu)
    cpu_time = time.perf_counter() - t0
    cpu_iterations = solver_cpu.getIterations()

    # --- Comparison ---
    max_diff = max(abs(u_cpu[i] - u_cuda[i]) for i in range(n))
    print(f"Grid: {nx}x{ny} = {n} points, t={final_t}")
    print(f"GPU iterations: {gpu_iterations}, time: {gpu_time:.3f}s")
    print(f"CPU iterations: {cpu_iterations}, time: {cpu_time:.3f}s")
    print(f"Speedup:  {cpu_time / gpu_time:.1f}x")
    print(f"Max diff CPU vs GPU: {max_diff:.2e}")


if __name__ == "__main__":
    main()

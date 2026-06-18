# mypy: disable-error-code="import-not-found, import-untyped, no-any-unimported, no-untyped-call, type-arg, unused-ignore"
# pyright: standard
# pyright: reportMissingImports=information
# pyright: reportIndexIssue=none

import warnings
from collections.abc import Callable
from typing import Any

import pytest

from pytnl import devices
from pytnl.containers import Vector
from pytnl.solvers import Euler, ODESolver

from .test_ODESolver import ADAPTIVE_METHODS, NON_ADAPTIVE_METHODS, _make_rhs

pytest.importorskip("pytnl._solvers_cuda")

pytestmark = pytest.mark.cuda


def _setup_heat_equation(
    n: int = 41,
) -> tuple[Any, Any, Callable[[float, float, Any, Any], None], float, float, float]:
    final_t = 0.05
    output_time_step = 0.005
    h = 1.0 / (n - 1)
    tau = 0.1 * h * h
    h_sqr_inv = 1.0 / (h * h)

    u_cpu = Vector[float](n)
    u_cuda = Vector[float, devices.Cuda](n)

    for i in range(n):
        x = i * h
        val = 1.0 if 0.4 <= x <= 0.6 else 0.0
        u_cpu[i] = val
        u_cuda[i] = val

    rhs = _make_rhs(n, h_sqr_inv)

    return u_cpu, u_cuda, rhs, tau, output_time_step, final_t


def _solve_heat_equation(method: type, adaptive: float | None = None) -> tuple[Any, Any, Any, Any]:
    u_cpu, u_cuda, rhs, tau, output_time_step, final_t = _setup_heat_equation()

    solver_cpu = ODESolver[method]()
    solver_cpu.setTau(tau)
    solver_cpu.setTime(0.0)
    if adaptive is not None:
        solver_cpu.setAdaptivity(adaptive)

    solver_cuda = ODESolver[method, devices.Cuda]()
    solver_cuda.setTau(tau)
    solver_cuda.setTime(0.0)
    if adaptive is not None:
        solver_cuda.setAdaptivity(adaptive)

    while solver_cpu.getTime() < final_t:
        stop_time = min(solver_cpu.getTime() + output_time_step, final_t)
        solver_cpu.setStopTime(stop_time)
        solver_cuda.setStopTime(stop_time)

        if not solver_cpu.solve(u_cpu, rhs):
            raise RuntimeError(f"CPU solver {method.__name__} did not converge")
        if not solver_cuda.solve(u_cuda, rhs):
            raise RuntimeError(f"CUDA solver {method.__name__} did not converge")

    return solver_cpu, solver_cuda, u_cpu, u_cuda


@pytest.mark.parametrize("method", NON_ADAPTIVE_METHODS)
def test_heat_equation_non_adaptive(method: type) -> None:
    """CUDA vs CPU comparison for non-adaptive methods — verifies identical results."""
    solver_cpu, solver_cuda, u_cpu, u_cuda = _solve_heat_equation(method)

    assert solver_cpu.getTime() == solver_cpu.getStopTime()
    assert solver_cuda.getTime() == solver_cuda.getStopTime()

    n = u_cpu.getSize()
    max_diff = max(abs(u_cpu[i] - u_cuda[i]) for i in range(n))
    assert max_diff < 1e-10, f"{method.__name__}: CPU vs CUDA max diff = {max_diff}"

    assert u_cuda[0] == 0.0, f"{method.__name__}: left boundary violated"
    assert u_cuda[n - 1] == 0.0, f"{method.__name__}: right boundary violated"


@pytest.mark.parametrize("method", ADAPTIVE_METHODS)
def test_heat_equation_adaptive(method: type) -> None:
    """CUDA vs CPU comparison for adaptive methods — verifies identical results."""
    solver_cpu, solver_cuda, u_cpu, u_cuda = _solve_heat_equation(method, adaptive=0.001)

    assert solver_cpu.getTime() == solver_cpu.getStopTime()
    assert solver_cuda.getTime() == solver_cuda.getStopTime()

    n = u_cpu.getSize()
    max_diff = max(abs(u_cpu[i] - u_cuda[i]) for i in range(n))
    assert max_diff < 1e-10, f"{method.__name__}: CPU vs CUDA max diff = {max_diff}"

    assert u_cuda[0] == 0.0, f"{method.__name__}: left boundary violated"
    assert u_cuda[n - 1] == 0.0, f"{method.__name__}: right boundary violated"


def test_solver_properties_cuda() -> None:
    """Verify getter/setter pairs and isStatic on ODESolver[Euler, devices.Cuda]."""
    solver = ODESolver[Euler, devices.Cuda]()

    solver.setTime(1.5)
    assert solver.getTime() == 1.5

    solver.setTau(0.01)
    assert solver.getTau() == 0.01

    solver.setStopTime(2.0)
    assert solver.getStopTime() == 2.0

    solver.setMaxTau(0.1)
    assert solver.getMaxTau() == 0.1

    solver.setAdaptivity(0.001)
    assert solver.getAdaptivity() == 0.001

    assert not ODESolver[Euler, devices.Cuda].isStatic()


def test_module_dispatch() -> None:
    """Verify ODESolver[Euler] resolves to Host module and ODESolver[Euler, Cuda] to CUDA module."""
    host_cls = ODESolver[Euler]
    cuda_cls = ODESolver[Euler, devices.Cuda]

    assert host_cls.__module__ == "pytnl._solvers"
    assert cuda_cls.__module__ == "pytnl._solvers_cuda"
    assert host_cls is not cuda_cls


def test_numba_cuda_jit_rhs() -> None:
    """ODESolver[Euler, devices.Cuda] with numba @cuda.jit RHS kernel via DLPack.

    This test exercises the actual GPU-accelerated RHS path: the solver passes
    CUDA VectorView objects to the Python callback, which launches a numba
    @cuda.jit kernel operating on the VectorView's GPU memory directly through
    the DLPack protocol — zero-copy.
    """
    nb_cuda = pytest.importorskip("numba.cuda")

    @nb_cuda.jit  # type: ignore[misc]
    def heat_rhs_kernel(
        u: Any,  # noqa: ANN401
        fu: Any,  # noqa: ANN401
        h_sqr_inv: float,
    ) -> None:
        i = nb_cuda.grid(1)
        n = u.size
        if i < n:
            if i == 0 or i == n - 1:
                fu[i] = 0.0
            else:
                fu[i] = h_sqr_inv * (u[i - 1] - 2.0 * u[i] + u[i + 1])

    threads_per_block = 256

    def make_rhs(n: int, h_sqr_inv: float) -> Callable[[float, float, Any, Any], None]:
        blocks = (n + threads_per_block - 1) // threads_per_block

        def rhs(
            t: float,
            tau: float,
            u_view: Any,  # noqa: ANN401
            fu_view: Any,  # noqa: ANN401
        ) -> None:
            heat_rhs_kernel[blocks, threads_per_block](u_view, fu_view, h_sqr_inv)

        return rhs

    n = 101
    h = 1.0 / (n - 1)
    tau = 0.1 * h * h
    final_t = 0.01
    h_sqr_inv = 1.0 / (h * h)

    u_cuda = Vector[float, devices.Cuda](n)
    for i in range(n):
        x = i * h
        u_cuda[i] = 1.0 if 0.4 <= x <= 0.6 else 0.0

    u_cpu = Vector[float](n)
    for i in range(n):
        x = i * h
        u_cpu[i] = 1.0 if 0.4 <= x <= 0.6 else 0.0

    # Warm up JIT compilation
    warm_u = Vector[float, devices.Cuda](n)
    warm_fu = Vector[float, devices.Cuda](n)
    warm_rhs = make_rhs(n, h_sqr_inv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warm_rhs(0.0, tau, warm_u.getView(), warm_fu.getView())
    nb_cuda.synchronize()

    # Solve on GPU with numba @cuda.jit RHS
    solver_gpu = ODESolver[Euler, devices.Cuda]()
    solver_gpu.setTau(tau)
    solver_gpu.setTime(0.0)
    solver_gpu.setStopTime(final_t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver_gpu.solve(u_cuda, make_rhs(n, h_sqr_inv))
    assert result, "CUDA solver with numba @cuda.jit RHS did not converge"

    # Solve on CPU for comparison
    solver_cpu = ODESolver[Euler]()
    solver_cpu.setTau(tau)
    solver_cpu.setTime(0.0)
    solver_cpu.setStopTime(final_t)
    cpu_result = solver_cpu.solve(u_cpu, _make_rhs(n, h_sqr_inv))
    assert cpu_result, "CPU solver did not converge"

    max_diff = max(abs(u_cpu[i] - u_cuda[i]) for i in range(n))
    assert max_diff < 1e-10, f"CPU vs numba-cuda GPU max diff = {max_diff}"

    assert u_cuda[0] == 0.0, "left boundary violated"
    assert u_cuda[n - 1] == 0.0, "right boundary violated"

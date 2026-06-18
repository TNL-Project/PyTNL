# mypy: disable-error-code="import-untyped,misc,no-untyped-def,no-untyped-usage,no-any-return,import-not-found"
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnknownMemberType=false, reportAssignmentType=false, reportCallIssue=false, reportIndexIssue=false, reportMissingImports=false
import time
from functools import partial
from typing import Any

from numba import jit

from pytnl.containers import Vector
from pytnl.solvers import DormandPrince, Euler, ODESolver


@jit(nopython=True)
def heat_rhs_numba(t: float, tau: float, u: Any, fu: Any, h_sqr_inv: float) -> None:  # noqa: ANN401
    """Compute the RHS of the 1D heat equation using central finite differences."""
    n = len(u)
    for i in range(1, n - 1):
        fu[i] = h_sqr_inv * (u[i - 1] - 2.0 * u[i] + u[i + 1])
    # Dirichlet boundary conditions
    fu[0] = 0.0
    fu[n - 1] = 0.0


def main() -> None:
    # 1D heat equation: du/dt = d^2u/dx^2
    n = 401
    final_t = 0.05
    output_time_step = 0.005
    h = 1.0 / (n - 1)
    tau = 0.5 * h * h
    h_sqr_inv = 1.0 / (h * h)
    adaptivity = 0.001

    def rhs_python(t: float, tau: float, u_view: Any, fu_view: Any) -> None:  # noqa: ANN401
        for i in range(1, n - 1):
            fu_view[i] = h_sqr_inv * (u_view[i - 1] - 2.0 * u_view[i] + u_view[i + 1])
        fu_view[0] = 0.0
        fu_view[n - 1] = 0.0

    # Numba-accelerated RHS via functools.partial — bakes in the constant h_sqr_inv
    rhs_numba = partial(heat_rhs_numba, h_sqr_inv=h_sqr_inv)

    # Initial condition: square pulse in [0.4, 0.6]
    def init(u: Any) -> None:  # noqa: ANN401
        for i in range(n):
            x = i * h
            if 0.4 <= x <= 0.6:
                u[i] = 1.0
            else:
                u[i] = 0.0

    def run_solver(rhs_func: Any, label: str) -> tuple[Any, float]:  # noqa: ANN401
        u = Vector[float](n)
        init(u)
        solver = ODESolver[Euler]()
        solver.setTau(tau)
        solver.setTime(0.0)
        start = time.perf_counter()
        while solver.getTime() < final_t:
            stop_time = min(solver.getTime() + output_time_step, final_t)
            solver.setStopTime(stop_time)
            solver.solve(u, rhs_func)
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.4f}s, max={max(u):.6f}")
        return u, elapsed

    # Warm up numba compilation (first call compiles, subsequent calls are fast)
    u_warm = Vector[float](n)
    fu_warm = Vector[float](n)
    rhs_numba(0.0, tau, u_warm.getView(), fu_warm.getView())

    print(f"Solving 1D heat equation (Euler, n={n}, final_t={final_t})")
    print()

    u_py, t_py = run_solver(rhs_python, "Pure Python RHS")
    u_nb, t_nb = run_solver(rhs_numba, "Numba JIT RHS   ")

    max_diff = max(abs(u_py[i] - u_nb[i]) for i in range(n))
    print(f"\nMax difference: {max_diff:.2e}")
    print(f"Speedup: {t_py / t_nb:.1f}x")

    # Also demonstrate with DormandPrince (adaptive)
    print()
    print(f"DormandPrince (adaptive, adaptivity={adaptivity}) with numba RHS:")

    u_dp = Vector[float](n)
    init(u_dp)
    solver_dp = ODESolver[DormandPrince]()
    solver_dp.setTau(tau)
    solver_dp.setTime(0.0)
    solver_dp.setAdaptivity(adaptivity)

    while solver_dp.getTime() < final_t:
        stop_time = min(solver_dp.getTime() + output_time_step, final_t)
        solver_dp.setStopTime(stop_time)
        solver_dp.solve(u_dp, rhs_numba)

    print(f"  t={solver_dp.getTime():.4f}, max={max(u_dp):.6f}")


if __name__ == "__main__":
    main()

from collections.abc import Callable
from typing import Any

import pytest

from pytnl.containers import Vector
from pytnl.solvers import (
    SSPRK3,
    BogackiShampin,
    CashKarp,
    DormandPrince,
    Euler,
    Fehlberg2,
    Fehlberg5,
    Heun2,
    Heun3,
    Kutta,
    KuttaMerson,
    Midpoint,
    ODESolver,
    OriginalRungeKutta,
    Ralston2,
    Ralston3,
    Ralston4,
    Rule38,
    VanDerHouwenWray,
)

# Non-adaptive methods (fixed step only)
NON_ADAPTIVE_METHODS = [
    Euler,
    OriginalRungeKutta,
    Midpoint,
    Heun3,
    Kutta,
    Ralston2,
    Ralston3,
    Ralston4,
    Rule38,
    SSPRK3,
    VanDerHouwenWray,
]

# Adaptive methods (support setAdaptivity)
ADAPTIVE_METHODS = [
    DormandPrince,
    CashKarp,
    KuttaMerson,
    Fehlberg2,
    Fehlberg5,
    BogackiShampin,
    Heun2,
]


def _make_rhs(n: int, h_sqr_inv: float) -> Callable[[float, float, Any, Any], None]:
    """Create the RHS function for the 1D heat equation du/dt = d^2u/dx^2.

    The signature matches what ODESolver.solve() expects:
    f(t, tau, u_view, fu_view) -> None
    """

    def rhs(
        t: float,
        tau: float,
        u_view: Any,  # noqa: ANN401
        fu_view: Any,  # noqa: ANN401
    ) -> None:
        for i in range(n):
            if i == 0 or i == n - 1:
                fu_view[i] = 0.0
            else:
                fu_view[i] = h_sqr_inv * (u_view[i - 1] - 2.0 * u_view[i] + u_view[i + 1])

    return rhs


def _setup_heat_equation(
    n: int = 41,
) -> tuple[Any, Callable[[float, float, Any, Any], None], float, float, float]:
    """Set up the 1D heat equation problem.

    Returns (solution_vector, rhs_function, tau, output_time_step, final_t).
    """
    final_t = 0.05
    output_time_step = 0.005
    h = 1.0 / (n - 1)
    tau = 0.1 * h * h
    h_sqr_inv = 1.0 / (h * h)

    u = Vector[float](n)

    # Initial condition: u[i]=1 for 0.4 <= x <= 0.6, else 0
    for i in range(n):
        x = i * h
        if 0.4 <= x <= 0.6:
            u[i] = 1.0
        else:
            u[i] = 0.0

    rhs = _make_rhs(n, h_sqr_inv)

    return u, rhs, tau, output_time_step, final_t


def _solve_heat_equation(method: type, adaptive: float | None = None) -> tuple[Any, Any]:
    """Solve the 1D heat equation with the given method.

    Returns (solver, solution_vector).
    """
    u, rhs, tau, output_time_step, final_t = _setup_heat_equation()

    solver = ODESolver[method]()
    solver.setTau(tau)
    solver.setTime(0.0)

    if adaptive is not None:
        solver.setAdaptivity(adaptive)

    while solver.getTime() < final_t:
        stop_time = min(solver.getTime() + output_time_step, final_t)
        solver.setStopTime(stop_time)

        if not solver.solve(u, rhs):
            raise RuntimeError(f"Solver {method.__name__} did not converge at time {solver.getTime()}")

    return solver, u


@pytest.mark.parametrize("method", NON_ADAPTIVE_METHODS)
def test_heat_equation(method: type) -> None:
    """1D heat equation solved with non-adaptive methods — verifies decay and boundary conditions."""
    solver, solution = _solve_heat_equation(method)

    assert solver.getTime() == solver.getStopTime(), f"{method.__name__}: solver did not reach final time"
    assert max(solution) < 1.0, f"{method.__name__}: initial peak of 1.0 did not decay (max={max(solution)})"
    n = solution.getSize()
    assert solution[0] == 0.0, f"{method.__name__}: left boundary condition violated"
    assert solution[n - 1] == 0.0, f"{method.__name__}: right boundary condition violated"


@pytest.mark.parametrize("method", ADAPTIVE_METHODS)
def test_heat_equation_adaptive(method: type) -> None:
    """1D heat equation solved with adaptive methods — verifies decay and boundary conditions."""
    solver, solution = _solve_heat_equation(method, adaptive=0.001)

    assert solver.getTime() == solver.getStopTime(), f"{method.__name__}: solver did not reach final time"
    assert max(solution) < 1.0, f"{method.__name__}: initial peak of 1.0 did not decay (max={max(solution)})"
    n = solution.getSize()
    assert solution[0] == 0.0, f"{method.__name__}: left boundary condition violated"
    assert solution[n - 1] == 0.0, f"{method.__name__}: right boundary condition violated"


def test_solver_properties() -> None:
    """Verify getter/setter pairs and isStatic on ODESolver[Euler]."""
    solver = ODESolver[Euler]()

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

    assert not ODESolver[Euler].isStatic()

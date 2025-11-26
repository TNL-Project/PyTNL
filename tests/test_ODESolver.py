from typing import TypeVar

import pytest

import pytnl._containers as _containers
import pytnl._solvers as _solvers

# ----------------------
# Configuration
# ----------------------

# Type aliases for vector types
type Vint = _containers.Vector_int
type Vfloat = _containers.Vector_float
type Vcomplex = _containers.Vector_complex

# Type aliases for vector view types
type Vint_view = _containers.VectorView_int
type Vfloat_view = _containers.VectorView_float
type Vcomplex_view = _containers.VectorView_complex

# Type variable constraining the solver types
S = TypeVar(
    "S",
    _solvers.ODESolver_Euler,
    _solvers.ODESolver_Euler,
)

# List of solver types to test
solver_types = S.__constraints__


def solve_heat_equation(solver_class: type[S], vector_class: type[Vfloat]) -> tuple[S, Vfloat]:
    # Parameters of the discretization
    final_t = 0.05
    output_time_step = 0.005
    n = 41
    h = 1.0 / (n - 1)
    tau = 0.1 * h * h
    h_sqr_inv = 1.0 / (h * h)

    # Initialize the solution vector
    u = vector_class(n)

    # Set initial condition
    for i in range(n):
        x = i * h
        if 0.4 <= x <= 0.6:
            u[i] = 1
        else:
            u[i] = 0

    # Setup solver
    solver = solver_class()
    solver.setTau(tau)
    solver.setTime(0)

    # Time loop
    while solver.getTime() < final_t:
        stop_time = min(solver.getTime() + output_time_step, final_t)
        solver.setStopTime(stop_time)

        def f(i: int, u_view: Vfloat_view, fu_view: Vfloat_view) -> None:
            if i == 0 or i == n - 1:  # boundary nodes
                fu_view[i] = 0
            else:  # interior nodes
                fu_view[i] = h_sqr_inv * (u_view[i - 1] - 2.0 * u_view[i] + u_view[i + 1])

        def time_stepping(t: float, tau_val: float, u_view: Vfloat_view, fu_view: Vfloat_view) -> None:
            for i in range(n):
                f(i, u_view, fu_view)

        if not solver.solve(u, time_stepping):
            raise RuntimeError(f"Solver did not converge at time {solver.getTime()}")

    return solver, u


@pytest.mark.parametrize("solver_type", solver_types)
def test_heat_equation(solver_type: type[S]) -> None:
    solver, solution = solve_heat_equation(solver_type, _containers.Vector_float)

    assert solver.getTime() == solver.getStopTime()

    # FIXME: convergence is not used by time stepping solvers (?!?)
    # assert solver.checkConvergence()
    # assert not solver.checkNextIteration()

    # TODO: verify the solution
    assert solution

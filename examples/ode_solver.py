from typing import Any

from pytnl.containers import Vector
from pytnl.solvers import DormandPrince, Euler, ODESolver


def main() -> None:
    # 1D heat equation: du/dt = d^2u/dx^2
    # Discretized with central finite differences
    n = 41
    final_t = 0.05
    output_time_step = 0.005
    h = 1.0 / (n - 1)
    tau = 0.5 * h * h
    h_sqr_inv = 1.0 / (h * h)
    adaptivity = 0.001

    def rhs(t: float, tau: float, u_view: Any, fu_view: Any) -> None:  # noqa: ANN401
        for i in range(n):
            if i == 0 or i == n - 1:
                fu_view[i] = 0.0
            else:
                fu_view[i] = h_sqr_inv * (u_view[i - 1] - 2.0 * u_view[i] + u_view[i + 1])

    # Initial condition: square pulse in [0.4, 0.6]
    u_euler = Vector[float](n)
    u_dp = Vector[float](n)

    for i in range(n):
        x = i * h
        if 0.4 <= x <= 0.6:
            u_euler[i] = 1.0
            u_dp[i] = 1.0
        else:
            u_euler[i] = 0.0
            u_dp[i] = 0.0

    # --- Euler (fixed step) ---
    solver_euler = ODESolver[Euler]()
    solver_euler.setTau(tau)
    solver_euler.setTime(0.0)

    print("Solving 1D heat equation (Euler, fixed step)")
    print(f"  n={n}, h={h:.6f}, tau={tau:.6f}, final_t={final_t}")
    print(f"  Initial: max={max(u_euler):.6f}, u[0]={u_euler[0]}, u[{n - 1}]={u_euler[n - 1]}")

    while solver_euler.getTime() < final_t:
        stop_time = min(solver_euler.getTime() + output_time_step, final_t)
        solver_euler.setStopTime(stop_time)
        solver_euler.solve(u_euler, rhs)

    print(f"  Euler:   t={solver_euler.getTime():.4f}, max={max(u_euler):.6f}, u[0]={u_euler[0]}, u[{n - 1}]={u_euler[n - 1]}")

    # --- DormandPrince (adaptive) ---
    solver_dp = ODESolver[DormandPrince]()
    solver_dp.setTau(tau)
    solver_dp.setTime(0.0)
    solver_dp.setAdaptivity(adaptivity)

    print("\nSolving 1D heat equation (DormandPrince, adaptive)")
    print(f"  n={n}, h={h:.6f}, tau={tau:.6f}, final_t={final_t}, adaptivity={adaptivity}")
    print(f"  Initial: max={max(u_dp):.6f}, u[0]={u_dp[0]}, u[{n - 1}]={u_dp[n - 1]}")

    while solver_dp.getTime() < final_t:
        stop_time = min(solver_dp.getTime() + output_time_step, final_t)
        solver_dp.setStopTime(stop_time)
        solver_dp.solve(u_dp, rhs)

    print(f"  DormandPrince: t={solver_dp.getTime():.4f}, max={max(u_dp):.6f}, u[0]={u_dp[0]}, u[{n - 1}]={u_dp[n - 1]}")

    print("\nDone — both solvers completed successfully.")


if __name__ == "__main__":
    main()

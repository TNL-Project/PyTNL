from __future__ import annotations

from typing import Any, overload

import pytnl._solvers
from pytnl._meta import CPPClassTemplate

__all__ = [
    "SSPRK3",
    "BogackiShampin",
    "CashKarp",
    "DormandPrince",
    "Euler",
    "ExplicitSolver",
    "Fehlberg2",
    "Fehlberg5",
    "Heun2",
    "Heun3",
    "IterativeSolver",
    "Kutta",
    "KuttaMerson",
    "Midpoint",
    "ODE1Solver",
    "ODE23Solver",
    "ODE45Solver",
    "ODESolver",
    "OriginalRungeKutta",
    "Ralston2",
    "Ralston3",
    "Ralston4",
    "Rule38",
    "VanDerHouwenWray",
]


class Euler:
    """First-order Euler method."""


class BogackiShampin:
    """Third-order Bogacki-Shampin method with adaptive step (Matlab ode23)."""


class DormandPrince:
    """Fifth-order Dormand-Prince method (Matlab ode45)."""


class CashKarp:
    """Fifth-order Cash-Karp method with adaptive step."""


class Fehlberg2:
    """Second-order Fehlberg method with adaptive step."""


class Fehlberg5:
    """Fifth-order Fehlberg method with adaptive step."""


class Heun2:
    """Second-order Heun method with adaptive step."""


class Heun3:
    """Third-order Heun method."""


class Kutta:
    """Third-order Kutta method."""


class KuttaMerson:
    """Fourth-order Runge-Kutta-Merson method with adaptive step."""


class Midpoint:
    """Second-order midpoint method."""


class OriginalRungeKutta:
    """Classic fourth-order Runge-Kutta method."""


class Ralston2:
    """Second-order Ralston method."""


class Ralston3:
    """Third-order Ralston method."""


class Ralston4:
    """Fourth-order Ralston method."""


class Rule38:
    """Fourth-order 3/8-rule Runge-Kutta method."""


class SSPRK3:
    """Third-order Strong Stability Preserving Runge-Kutta method."""


class VanDerHouwenWray:
    """Third-order Van der Houwen-Wray method."""


class _ODESolverMeta(CPPClassTemplate):  # type: ignore[no-any-unimported, unused-ignore]
    _cpp_module = pytnl._solvers
    _class_prefix = "ODESolver"
    _template_parameters = (("method", type),)

    @overload
    def __getitem__(self, key: type[BogackiShampin], /) -> type[pytnl._solvers.ODESolver_BogackiShampin]: ...

    @overload
    def __getitem__(self, key: type[CashKarp], /) -> type[pytnl._solvers.ODESolver_CashKarp]: ...

    @overload
    def __getitem__(self, key: type[DormandPrince], /) -> type[pytnl._solvers.ODESolver_DormandPrince]: ...

    @overload
    def __getitem__(self, key: type[Euler], /) -> type[pytnl._solvers.ODESolver_Euler]: ...

    @overload
    def __getitem__(self, key: type[Fehlberg2], /) -> type[pytnl._solvers.ODESolver_Fehlberg2]: ...

    @overload
    def __getitem__(self, key: type[Fehlberg5], /) -> type[pytnl._solvers.ODESolver_Fehlberg5]: ...

    @overload
    def __getitem__(self, key: type[Heun2], /) -> type[pytnl._solvers.ODESolver_Heun2]: ...

    @overload
    def __getitem__(self, key: type[Heun3], /) -> type[pytnl._solvers.ODESolver_Heun3]: ...

    @overload
    def __getitem__(self, key: type[Kutta], /) -> type[pytnl._solvers.ODESolver_Kutta]: ...

    @overload
    def __getitem__(self, key: type[KuttaMerson], /) -> type[pytnl._solvers.ODESolver_KuttaMerson]: ...

    @overload
    def __getitem__(self, key: type[Midpoint], /) -> type[pytnl._solvers.ODESolver_Midpoint]: ...

    @overload
    def __getitem__(self, key: type[OriginalRungeKutta], /) -> type[pytnl._solvers.ODESolver_OriginalRungeKutta]: ...

    @overload
    def __getitem__(self, key: type[Ralston2], /) -> type[pytnl._solvers.ODESolver_Ralston2]: ...

    @overload
    def __getitem__(self, key: type[Ralston3], /) -> type[pytnl._solvers.ODESolver_Ralston3]: ...

    @overload
    def __getitem__(self, key: type[Ralston4], /) -> type[pytnl._solvers.ODESolver_Ralston4]: ...

    @overload
    def __getitem__(self, key: type[Rule38], /) -> type[pytnl._solvers.ODESolver_Rule38]: ...

    @overload
    def __getitem__(self, key: type[SSPRK3], /) -> type[pytnl._solvers.ODESolver_SSPRK3]: ...

    @overload
    def __getitem__(self, key: type[VanDerHouwenWray], /) -> type[pytnl._solvers.ODESolver_VanDerHouwenWray]: ...

    def __getitem__(self, key: type[Any], /) -> type[Any]:
        return self._get_cpp_class((key,))  # type: ignore[no-any-unimported, unused-ignore]


class ODESolver(metaclass=_ODESolverMeta):
    """
    Allows `ODESolver[Method]` syntax to resolve to the appropriate C++
    `ODESolver` class.

    This class provides a Python interface to C++ ODE solvers with a specific
    numerical method.

    Examples:
    - `ODESolver[Euler]` → `_solvers.ODESolver_Euler`
    - `ODESolver[DormandPrince]` → `_solvers.ODESolver_DormandPrince`
    """


IterativeSolver = pytnl._solvers.IterativeSolver_float_int
ExplicitSolver = pytnl._solvers.ExplicitSolver_float_int

# Convenience aliases matching Matlab ODE solver names
ODE1Solver = Euler
ODE23Solver = BogackiShampin
ODE45Solver = DormandPrince

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pytnl._solvers
import pytnl.devices
from pytnl._meta import CPPClassTemplate

if TYPE_CHECKING:
    # This is an optional module - at runtime it is lazy-imported in
    # `CPPClassTemplate`, for type checking there must be the import statement.
    import pytnl._solvers_cuda as _solvers_cuda  # type: ignore[import-not-found, unused-ignore]

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
    _template_parameters = (
        ("method", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(
        self, key: type[BogackiShampin] | tuple[type[BogackiShampin], type[pytnl.devices.Host]], /
    ) -> type[pytnl._solvers.ODESolver_BogackiShampin]: ...

    @overload
    def __getitem__(self, key: tuple[type[BogackiShampin], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_BogackiShampin]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[CashKarp] | tuple[type[CashKarp], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_CashKarp]: ...

    @overload
    def __getitem__(self, key: tuple[type[CashKarp], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_CashKarp]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self, key: type[DormandPrince] | tuple[type[DormandPrince], type[pytnl.devices.Host]], /
    ) -> type[pytnl._solvers.ODESolver_DormandPrince]: ...

    @overload
    def __getitem__(self, key: tuple[type[DormandPrince], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_DormandPrince]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Euler] | tuple[type[Euler], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Euler]: ...

    @overload
    def __getitem__(self, key: tuple[type[Euler], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Euler]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Fehlberg2] | tuple[type[Fehlberg2], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Fehlberg2]: ...

    @overload
    def __getitem__(self, key: tuple[type[Fehlberg2], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Fehlberg2]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Fehlberg5] | tuple[type[Fehlberg5], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Fehlberg5]: ...

    @overload
    def __getitem__(self, key: tuple[type[Fehlberg5], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Fehlberg5]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Heun2] | tuple[type[Heun2], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Heun2]: ...

    @overload
    def __getitem__(self, key: tuple[type[Heun2], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Heun2]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Heun3] | tuple[type[Heun3], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Heun3]: ...

    @overload
    def __getitem__(self, key: tuple[type[Heun3], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Heun3]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Kutta] | tuple[type[Kutta], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Kutta]: ...

    @overload
    def __getitem__(self, key: tuple[type[Kutta], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Kutta]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[KuttaMerson] | tuple[type[KuttaMerson], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_KuttaMerson]: ...

    @overload
    def __getitem__(self, key: tuple[type[KuttaMerson], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_KuttaMerson]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Midpoint] | tuple[type[Midpoint], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Midpoint]: ...

    @overload
    def __getitem__(self, key: tuple[type[Midpoint], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Midpoint]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self, key: type[OriginalRungeKutta] | tuple[type[OriginalRungeKutta], type[pytnl.devices.Host]], /
    ) -> type[pytnl._solvers.ODESolver_OriginalRungeKutta]: ...

    @overload
    def __getitem__(self, key: tuple[type[OriginalRungeKutta], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_OriginalRungeKutta]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Ralston2] | tuple[type[Ralston2], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Ralston2]: ...

    @overload
    def __getitem__(self, key: tuple[type[Ralston2], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Ralston2]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Ralston3] | tuple[type[Ralston3], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Ralston3]: ...

    @overload
    def __getitem__(self, key: tuple[type[Ralston3], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Ralston3]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Ralston4] | tuple[type[Ralston4], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Ralston4]: ...

    @overload
    def __getitem__(self, key: tuple[type[Ralston4], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Ralston4]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[Rule38] | tuple[type[Rule38], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_Rule38]: ...

    @overload
    def __getitem__(self, key: tuple[type[Rule38], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_Rule38]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(self, key: type[SSPRK3] | tuple[type[SSPRK3], type[pytnl.devices.Host]], /) -> type[pytnl._solvers.ODESolver_SSPRK3]: ...

    @overload
    def __getitem__(self, key: tuple[type[SSPRK3], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_SSPRK3]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self, key: type[VanDerHouwenWray] | tuple[type[VanDerHouwenWray], type[pytnl.devices.Host]], /
    ) -> type[pytnl._solvers.ODESolver_VanDerHouwenWray]: ...

    @overload
    def __getitem__(self, key: tuple[type[VanDerHouwenWray], type[pytnl.devices.Cuda]], /) -> type[_solvers_cuda.ODESolver_VanDerHouwenWray]: ...  # type: ignore[no-any-unimported, unused-ignore]  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(self, key: type[Any] | tuple[type[Any], type[Any]], /) -> type[Any]:
        if isinstance(key, tuple):
            items = key
        else:
            items = (key, pytnl.devices.Host)
        return self._get_cpp_class(items)  # type: ignore[no-any-unimported, unused-ignore]


class ODESolver(metaclass=_ODESolverMeta):
    """
    Allows `ODESolver[Method]` or `ODESolver[Method, DeviceType]` syntax to
    resolve to the appropriate C++ `ODESolver` class.

    This class provides a Python interface to C++ ODE solvers with a specific
    numerical method and optional device type.

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.

    Examples:
    - `ODESolver[Euler]` → `_solvers.ODESolver_Euler`
    - `ODESolver[Euler, devices.Host]` → `_solvers.ODESolver_Euler`
    - `ODESolver[DormandPrince, devices.Cuda]` → `_solvers_cuda.ODESolver_DormandPrince`
    """


IterativeSolver = pytnl._solvers.IterativeSolver_float_int
ExplicitSolver = pytnl._solvers.ExplicitSolver_float_int

# Convenience aliases matching Matlab ODE solver names
ODE1Solver = Euler
ODE23Solver = BogackiShampin
ODE45Solver = DormandPrince

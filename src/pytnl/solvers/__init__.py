from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import pytnl._solvers
import pytnl.devices
from pytnl._meta import CPPClassTemplate
from pytnl._solvers import ode_methods

if TYPE_CHECKING:
    # This is an optional module - at runtime it is lazy-imported in
    # `CPPClassTemplate`, for type checking there must be the import statement.
    import pytnl._solvers_cuda as _solvers_cuda  # type: ignore[import-not-found, unused-ignore]

__all__ = [
    "ExplicitSolver",
    "IterativeSolver",
    "ODESolver",
    "ode_methods",
]


class _ODESolverMeta(CPPClassTemplate):
    _cpp_module = pytnl._solvers
    _class_prefix = "ODESolver"
    _template_parameters = (
        ("method", type),
        ("device_type", type),
    )
    _device_parameter = "device_type"

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.BogackiShampin] | tuple[type[ode_methods.BogackiShampin], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_BogackiShampin]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.BogackiShampin], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_BogackiShampin]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.CashKarp] | tuple[type[ode_methods.CashKarp], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_CashKarp]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.CashKarp], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_CashKarp]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.DormandPrince] | tuple[type[ode_methods.DormandPrince], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_DormandPrince]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.DormandPrince], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_DormandPrince]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Euler] | tuple[type[ode_methods.Euler], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Euler]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Euler], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Euler]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Fehlberg2] | tuple[type[ode_methods.Fehlberg2], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Fehlberg2]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Fehlberg2], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Fehlberg2]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Fehlberg5] | tuple[type[ode_methods.Fehlberg5], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Fehlberg5]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Fehlberg5], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Fehlberg5]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Heun2] | tuple[type[ode_methods.Heun2], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Heun2]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Heun2], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Heun2]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Heun3] | tuple[type[ode_methods.Heun3], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Heun3]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Heun3], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Heun3]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Kutta] | tuple[type[ode_methods.Kutta], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Kutta]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Kutta], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Kutta]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.KuttaMerson] | tuple[type[ode_methods.KuttaMerson], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_KuttaMerson]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.KuttaMerson], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_KuttaMerson]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Midpoint] | tuple[type[ode_methods.Midpoint], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Midpoint]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Midpoint], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Midpoint]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.OriginalRungeKutta] | tuple[type[ode_methods.OriginalRungeKutta], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_OriginalRungeKutta]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.OriginalRungeKutta], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_OriginalRungeKutta]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Ralston2] | tuple[type[ode_methods.Ralston2], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Ralston2]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Ralston2], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Ralston2]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Ralston3] | tuple[type[ode_methods.Ralston3], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Ralston3]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Ralston3], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Ralston3]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Ralston4] | tuple[type[ode_methods.Ralston4], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Ralston4]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Ralston4], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Ralston4]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.Rule38] | tuple[type[ode_methods.Rule38], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_Rule38]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.Rule38], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_Rule38]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.SSPRK3] | tuple[type[ode_methods.SSPRK3], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_SSPRK3]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.SSPRK3], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_SSPRK3]: ...  # pyright: ignore[reportUnknownMemberType]

    @overload
    def __getitem__(
        self,
        key: type[ode_methods.VanDerHouwenWray] | tuple[type[ode_methods.VanDerHouwenWray], type[pytnl.devices.Host]],
        /,
    ) -> type[pytnl._solvers.ODESolver_VanDerHouwenWray]: ...

    @overload
    def __getitem__(  # type: ignore[no-any-unimported, unused-ignore]
        self,
        key: tuple[type[ode_methods.VanDerHouwenWray], type[pytnl.devices.Cuda]],
        /,
    ) -> type[_solvers_cuda.ODESolver_VanDerHouwenWray]: ...  # pyright: ignore[reportUnknownMemberType]

    def __getitem__(self, key: type[Any] | tuple[type[Any], type[Any]], /) -> type[Any]:
        if isinstance(key, tuple):
            items = key
        else:
            items = (key, pytnl.devices.Host)
        return self._get_cpp_class(items)


class ODESolver(metaclass=_ODESolverMeta):
    """
    Allows `ODESolver[Method]` or `ODESolver[Method, DeviceType]` syntax to
    resolve to the appropriate C++ `ODESolver` class.

    This class provides a Python interface to C++ ODE solvers with a specific
    numerical method and optional device type.

    The `device_type` argument is optional and defaults to `pytnl.devices.Host`.

    Examples:
    - `ODESolver[ode_methods.Euler]` → `_solvers.ODESolver_Euler`
    - `ODESolver[ode_methods.Euler, devices.Host]` → `_solvers.ODESolver_Euler`
    - `ODESolver[ode_methods.DormandPrince, devices.Cuda]` → `_solvers_cuda.ODESolver_DormandPrince`
    """


IterativeSolver = pytnl._solvers.IterativeSolver_float_int
ExplicitSolver = pytnl._solvers.ExplicitSolver_float_int

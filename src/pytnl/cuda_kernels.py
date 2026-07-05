"""
Compile Python functions as numba CUDA kernels for TNL CUDA arrays.

The ``@compile_device_func`` decorator marks a Python function for CUDA
execution via numba-cuda-mlir. The decorated function captures TNL CUDA arrays
via ``__cuda_array_interface__`` from the enclosing scope and receives
multi-dimensional indices, mirroring the host ``forAll``/``forInterior``/
``forBoundary`` interface::

    @compile_device_func
    def setter(*idx: int) -> None:
        a[idx] += 1

    a.forAll(setter)
"""

# pyright: reportUnknownMemberType=none, reportUnknownVariableType=none, reportUnknownArgumentType=none

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "CompiledDeviceFunc",
    "compile_device_func",
    "launch_array_for_elements",
    "launch_ndarray_for_all",
    "launch_ndarray_for_boundary",
    "launch_ndarray_for_interior",
]


def _collect_load_names(body: list[ast.stmt]) -> set[str]:
    """Return the set of identifiers used in Load context within *body*."""
    names: set[str] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                names.add(node.id)
    return names


class CompiledDeviceFunc:
    """A numba CUDA kernel compiled from a Python function.

    Created by the ``@compile_device_func`` decorator. The decorated function
    captures TNL CUDA arrays via ``__cuda_array_interface__`` and receives
    multi-dimensional indices::

        @compile_device_func
        def setter(*idx: int) -> None:
            a[idx] += 1

        a.forAll(setter)

    The setter may also use explicit index arguments instead of ``*idx``::

        @compile_device_func
        def setter(i: int, j: int) -> None:
            a[i, j] += 1
    """

    def __init__(self, func: Callable[..., Any], name: str | None = None) -> None:
        self._func = func
        self._name: str = name or str(getattr(func, "__name__", "user_func"))
        self._kernel_cache: dict[tuple[int, str], tuple[Any, list[str]]] = {}
        self.__wrapped__ = func
        self.__name__ = self._name
        self.__doc__ = str(getattr(func, "__doc__", None) or "")

    @property
    def name(self) -> str:
        return self._name

    def get_kernel(self, dim: int, mode: str) -> tuple[Any, list[str]]:
        """Return a cached ``(kernel, array_arg_names)`` tuple for the given dimension and mode.

        The kernel captures TNL CUDA arrays as parameters (not globals) to
        avoid numba ``Global`` IR nodes that would keep nanobind instances
        alive past shutdown. ``array_arg_names`` is the ordered list of CUDA
        array names that must be passed as positional arguments after
        ``sizes`` when launching.

        Args:
            dim: Array dimension (1, 2, or 3)
            mode: ``"all"``, ``"interior"``, ``"boundary"``, or ``"elements"``
                (1D only, used by :meth:`Array.forElements`)
        """
        key = (dim, mode)
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        result = self._generate_kernel(dim, mode)
        self._kernel_cache[key] = result
        return result

    def resolve_array_args(self, array_arg_names: list[str]) -> list[Any]:
        """Resolve current values of captured CUDA arrays from the wrapped function's scope."""
        func = self._func
        closure_map: dict[str, Any] = {}
        if func.__closure__:  # type: ignore[attr-defined]
            closure_map = dict(
                zip(
                    func.__code__.co_freevars,  # type: ignore[attr-defined]
                    (cell.cell_contents for cell in func.__closure__),  # type: ignore[attr-defined]
                )
            )
        args: list[Any] = []
        for name in array_arg_names:
            if name in closure_map:
                args.append(closure_map[name])
            else:
                args.append(func.__globals__[name])  # type: ignore[attr-defined]
        return args

    def _generate_kernel(self, dim: int, mode: str) -> tuple[Any, list[str]]:
        """Generate and compile a ``@cuda.jit`` kernel via ``exec``.

        The user's function body is inlined into the kernel. Index variables
        (``*idx`` or explicit args) are initialized from CUDA grid coordinates.
        CUDA-array-like objects captured from the setter's scope are passed as
        kernel parameters (not globals) to avoid numba ``Global`` IR nodes that
        keep nanobind instances alive past shutdown.

        Returns:
            A ``(kernel, array_arg_names)`` tuple.
        """
        from numba_cuda_mlir import cuda as nb_cuda  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415

        if mode == "boundary" and dim > 3:
            raise ValueError(f"forBoundary is not supported for dimension > 3 (got dim={dim}). Only forAll and forInterior support arbitrary dimensions.")

        if mode == "elements" and dim != 1:
            raise ValueError(f"forElements mode is only supported for 1D arrays (got dim={dim}).")

        user_func = self._func
        vararg_name, regular_args, body_src, closure_vars, array_arg_names = self._extract_func_setup(user_func)

        sizes_name = "_pytnl_sizes"
        while sizes_name in array_arg_names:
            sizes_name += "_"

        if mode == "elements":
            kernel_src = self._build_kernel_src_1d_elements(
                vararg_name,
                regular_args,
                body_src,
                array_arg_names,
            )
        elif dim <= 3:
            kernel_src = self._build_kernel_src_low_dim(
                dim,
                mode,
                vararg_name,
                regular_args,
                body_src,
                sizes_name,
                array_arg_names,
            )
        else:
            kernel_src = self._build_kernel_src_high_dim(
                dim,
                mode,
                vararg_name,
                regular_args,
                body_src,
                sizes_name,
                array_arg_names,
            )

        namespace: dict[str, Any] = {**user_func.__globals__, **closure_vars, "nb_cuda": nb_cuda}  # type: ignore[attr-defined]
        for name in array_arg_names:
            namespace.pop(name, None)

        exec(kernel_src, namespace)
        return namespace["_kernel"], array_arg_names

    @staticmethod
    def _extract_func_setup(user_func: Callable[..., Any]) -> tuple[str | None, list[str], str, dict[str, Any], list[str]]:
        """Extract AST, closure, and array-arg info from the user function."""
        src = textwrap.dedent(inspect.getsource(user_func))
        tree = ast.parse(src)
        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise RuntimeError(f"Expected FunctionDef, got {type(func_def).__name__}")

        vararg_name: str | None = func_def.args.vararg.arg if func_def.args.vararg else None
        regular_args = [a.arg for a in func_def.args.args]

        body_lines = [ast.unparse(stmt) for stmt in func_def.body]
        body_src = "\n".join(body_lines)

        closure_vars: dict[str, Any] = {}
        if user_func.__closure__:  # type: ignore[attr-defined]
            for name, cell in zip(user_func.__code__.co_freevars, user_func.__closure__):  # type: ignore[attr-defined]
                closure_vars[name] = cell.cell_contents

        used_names = _collect_load_names(func_def.body)
        array_arg_names: list[str] = []
        for name, value in {**user_func.__globals__, **closure_vars}.items():  # type: ignore[attr-defined]
            if name in used_names and hasattr(value, "__cuda_array_interface__"):
                array_arg_names.append(name)

        return vararg_name, regular_args, body_src, closure_vars, array_arg_names

    @staticmethod
    def _build_kernel_src_low_dim(
        dim: int,
        mode: str,
        vararg_name: str | None,
        regular_args: list[str],
        body_src: str,
        sizes_name: str,
        array_param_names: list[str],
    ) -> str:
        # Generate dimension-specific grid, bounds, and boundary check code
        if dim == 1:
            grid_code = "i = nb_cuda.grid(1)"
            bounds = f"i < {sizes_name}[0]"
            idx_vars = "i"
            idx_tuple = "(i,)"
            boundary = f"i == 0 or i == {sizes_name}[0] - 1"
        elif dim == 2:
            grid_code = "i, j = nb_cuda.grid(2)"
            bounds = f"i < {sizes_name}[0] and j < {sizes_name}[1]"
            idx_vars = "i, j"
            idx_tuple = "(i, j)"
            boundary = f"i == 0 or i == {sizes_name}[0] - 1 or j == 0 or j == {sizes_name}[1] - 1"
        else:
            grid_code = "i, j, k = nb_cuda.grid(3)"
            bounds = f"i < {sizes_name}[0] and j < {sizes_name}[1] and k < {sizes_name}[2]"
            idx_vars = "i, j, k"
            idx_tuple = "(i, j, k)"
            boundary = f"i == 0 or i == {sizes_name}[0] - 1 or j == 0 or j == {sizes_name}[1] - 1 or k == 0 or k == {sizes_name}[2] - 1"

        if mode == "all":
            filter_code = ""
        elif mode == "interior":
            filter_code = f"    if ({boundary}):\n        return\n"
        else:  # boundary
            filter_code = f"    if not ({boundary}):\n        return\n"

        # Initialize the index variable(s) from grid coordinates
        if vararg_name:
            idx_init = f"    {vararg_name} = {idx_tuple}\n"
        elif regular_args:
            idx_init = f"    {', '.join(regular_args)} = {idx_vars}\n"
        else:
            idx_init = ""

        body_indented = textwrap.indent(body_src, "    ")

        params = ", ".join([sizes_name, *array_param_names])
        return f"""@nb_cuda.jit
def _kernel({params}):
    {grid_code}
    if not ({bounds}):
        return
{filter_code}{idx_init}{body_indented}
"""

    @staticmethod
    def _build_kernel_src_high_dim(
        dim: int,
        mode: str,
        vararg_name: str | None,
        regular_args: list[str],
        body_src: str,
        sizes_name: str,
        array_param_names: list[str],
    ) -> str:
        # dim > 3: 3D grid for last 3 dims + sequential loops for first (dim-3) dims
        # i (x, fastest) -> dim N-1, j (y) -> dim N-2, k (z, slowest) -> dim N-3
        grid_code = "i, j, k = nb_cuda.grid(3)"

        if mode == "all":
            bounds = f"i < {sizes_name}[{dim - 1}] and j < {sizes_name}[{dim - 2}] and k < {sizes_name}[{dim - 3}]"
        else:  # interior
            bounds = f"1 <= i < {sizes_name}[{dim - 1}] - 1 and 1 <= j < {sizes_name}[{dim - 2}] - 1 and 1 <= k < {sizes_name}[{dim - 3}] - 1"

        # Generate sequential loops for dims 0 to dim-4
        seq_dims = list(range(dim - 3))
        seq_vars = [f"d{d}" for d in seq_dims]

        loop_code = ""
        for level, d in enumerate(seq_dims):
            indent = "    " * (1 + level)
            if mode == "all":
                loop_code += f"{indent}for d{d} in range({sizes_name}[{d}]):\n"
            else:  # interior
                loop_code += f"{indent}for d{d} in range(1, {sizes_name}[{d}] - 1):\n"

        # All index variables: sequential vars + parallel vars (k->N-3, j->N-2, i->N-1)
        all_vars = [*seq_vars, "k", "j", "i"]
        idx_tuple = "(" + ", ".join(all_vars) + ")"
        idx_vars_str = ", ".join(all_vars)

        # Initialize index variable(s) at the inner loop level
        inner_indent = "    " * (1 + len(seq_dims))
        if vararg_name:
            idx_init = f"{inner_indent}{vararg_name} = {idx_tuple}\n"
        elif regular_args:
            idx_init = f"{inner_indent}{', '.join(regular_args)} = {idx_vars_str}\n"
        else:
            idx_init = ""

        body_indented = textwrap.indent(body_src, inner_indent)

        params = ", ".join([sizes_name, *array_param_names])
        return f"""@nb_cuda.jit
def _kernel({params}):
    {grid_code}
    if not ({bounds}):
        return
{loop_code}{idx_init}{body_indented}
"""

    @staticmethod
    def _build_kernel_src_1d_elements(
        vararg_name: str | None,
        regular_args: list[str],
        body_src: str,
        array_param_names: list[str],
    ) -> str:
        begin_name = "_pytnl_begin"
        end_name = "_pytnl_end"
        while begin_name in array_param_names:
            begin_name += "_"
        while end_name in array_param_names:
            end_name += "_"

        if vararg_name:
            idx_init = f"    {vararg_name} = (i,)\n"
        elif regular_args:
            idx_init = f"    {regular_args[0]} = i\n"
        else:
            idx_init = ""

        body_indented = textwrap.indent(body_src, "    ")
        params = ", ".join([begin_name, end_name, *array_param_names])
        return f"""@nb_cuda.jit
def _kernel({params}):
    i = {begin_name} + nb_cuda.grid(1)
    if not (i < {end_name}):
        return
{idx_init}{body_indented}
"""


def compile_device_func(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
) -> CompiledDeviceFunc | Callable[[Callable[..., Any]], CompiledDeviceFunc]:
    """Decorator to mark a Python function for CUDA execution via numba.

    Usage (bare decorator)::

        @compile_device_func
        def setter(*idx: int) -> None:
            a[idx] += 1

        a.forAll(setter)

    Or with custom name::

        @compile_device_func(name="my_func")
        def setter(*idx: int) -> None:
            a[idx] += 1

    The setter captures TNL CUDA arrays via ``__cuda_array_interface__``
    from the enclosing scope.
    """
    if func is not None:
        return CompiledDeviceFunc(func, name=None)

    def decorator(f: Callable[..., Any]) -> CompiledDeviceFunc:
        return CompiledDeviceFunc(f, name=name)

    return decorator


def _launch_compiled_kernel(array: Any, compiled: CompiledDeviceFunc, mode: str) -> None:  # noqa: ANN401
    """Launch a ``CompiledDeviceFunc`` kernel over ``array`` in the given mode."""
    from numba_cuda_mlir import cuda as nb_cuda  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415

    dim = type(array).getDimension()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    sizes = array.getSizes()
    kernel, array_arg_names = compiled.get_kernel(dim, mode)  # pyright: ignore[reportUnknownArgumentType]
    array_args = compiled.resolve_array_args(array_arg_names)
    block_size = {1: 256, 2: 16, 3: 8}.get(dim, 8)  # pyright: ignore[reportUnknownArgumentType]
    grid_dims: tuple[int, ...] = tuple((s + block_size - 1) // block_size for s in sizes)
    if dim == 1:
        grid = (grid_dims[0], 1, 1)
        block = (block_size, 1, 1)
    elif dim == 2:
        grid = (grid_dims[1], grid_dims[0], 1)
        block = (block_size, block_size, 1)
    else:
        grid = (grid_dims[dim - 1], grid_dims[dim - 2], grid_dims[dim - 3])  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
        block = (block_size, block_size, block_size)
    kernel[grid, block](sizes, *array_args)  # pyright: ignore[reportUnknownMemberType]
    nb_cuda.synchronize()  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]


def _launch_raw_kernel(array: Any, kernel: Any) -> None:  # noqa: ANN401
    """Launch a raw ``@cuda.jit`` kernel over ``array``.

    The kernel must have signature ``(storage, sizes, strides)`` and compute
    the flat storage index itself.
    """
    from numba_cuda_mlir import cuda as nb_cuda  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415

    storage = array.getStorageArrayView()
    sizes = array.getSizes()
    strides = array.getStrides()
    dim = type(array).getDimension()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    tpb: Any
    bpg: Any
    if dim == 1:
        tpb = 256
        bpg = (sizes[0] + tpb - 1) // tpb
    elif dim == 2:
        tpb = (16, 16)
        bpg = ((sizes[0] + tpb[0] - 1) // tpb[0], (sizes[1] + tpb[1] - 1) // tpb[1])
    else:
        tpb = (8, 8, 8)
        bpg = (
            (sizes[0] + tpb[0] - 1) // tpb[0],
            (sizes[1] + tpb[1] - 1) // tpb[1],
            (sizes[2] + tpb[2] - 1) // tpb[2],
        )

    kernel[bpg, tpb](storage, sizes, strides)
    nb_cuda.synchronize()  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]


def launch_ndarray_for_all(array: Any, kernel: Any) -> None:  # noqa: ANN401
    """Launch ``kernel`` over all elements of ``array``.

    Accepts either a :class:`CompiledDeviceFunc` or a raw ``@cuda.jit`` kernel.
    """
    if isinstance(kernel, CompiledDeviceFunc):
        _launch_compiled_kernel(array, kernel, "all")
    else:
        _launch_raw_kernel(array, kernel)


def launch_ndarray_for_interior(array: Any, kernel: Any) -> None:  # noqa: ANN401
    """Launch ``kernel`` over interior elements of ``array``.

    Requires a :class:`CompiledDeviceFunc` — boundary elements are skipped
    automatically by the generated kernel.
    """
    if not isinstance(kernel, CompiledDeviceFunc):
        raise TypeError("forInterior requires a CompiledDeviceFunc (from @compile_device_func)")
    _launch_compiled_kernel(array, kernel, "interior")


def launch_ndarray_for_boundary(array: Any, kernel: Any) -> None:  # noqa: ANN401
    """Launch ``kernel`` over boundary elements of ``array``.

    Requires a :class:`CompiledDeviceFunc` — only boundary elements are
    processed automatically by the generated kernel.
    """
    if not isinstance(kernel, CompiledDeviceFunc):
        raise TypeError("forBoundary requires a CompiledDeviceFunc (from @compile_device_func)")
    _launch_compiled_kernel(array, kernel, "boundary")


def _launch_compiled_kernel_1d_elements(
    array: Any,  # noqa: ANN401
    compiled: CompiledDeviceFunc,
    begin: int,
    end: int,
) -> None:
    from numba_cuda_mlir import cuda as nb_cuda  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415

    if end == 0:
        end = len(array)
    kernel, array_arg_names = compiled.get_kernel(1, "elements")
    array_args = compiled.resolve_array_args(array_arg_names)
    n = end - begin
    tpb = 256
    bpg = (n + tpb - 1) // tpb
    if bpg < 1:
        bpg = 1
    kernel[bpg, tpb](begin, end, *array_args)  # pyright: ignore[reportUnknownMemberType]
    nb_cuda.synchronize()  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]


def _launch_raw_kernel_1d_elements(
    array: Any,  # noqa: ANN401
    kernel: Any,  # noqa: ANN401
    begin: int,
    end: int,
) -> None:
    from numba_cuda_mlir import cuda as nb_cuda  # type: ignore[import-not-found, unused-ignore]  # noqa: PLC0415

    if end == 0:
        end = len(array)
    n = end - begin
    tpb = 256
    bpg = (n + tpb - 1) // tpb
    if bpg < 1:
        bpg = 1
    kernel[bpg, tpb](array, begin, end)  # pyright: ignore[reportUnknownMemberType]
    nb_cuda.synchronize()  # type: ignore[attr-defined, unused-ignore]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]


def launch_array_for_elements(array: Any, begin: int, end: int, kernel: Any) -> None:  # noqa: ANN401
    """Launch ``kernel`` over elements in ``[begin, end)`` of 1D ``array``.

    For a :class:`CompiledDeviceFunc`, the setter receives the linear index
    ``i`` (in ``[begin, end)``) and accesses captured arrays directly.
    For a raw ``@cuda.jit`` kernel, the kernel is called as
    ``kernel(array, begin, end)`` and must compute ``i = begin + cuda.grid(1)``
    itself.
    """
    if isinstance(kernel, CompiledDeviceFunc):
        _launch_compiled_kernel_1d_elements(array, kernel, begin, end)
    else:
        _launch_raw_kernel_1d_elements(array, kernel, begin, end)

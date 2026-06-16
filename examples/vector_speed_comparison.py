import argparse
import math
import time
from collections.abc import Callable

import numpy as np

from pytnl._containers import Vector_float
from pytnl.containers import Vector

DEFAULT_SIZE_ADD = 1_000_000
DEFAULT_SIZE_DOT = 1_000
DEFAULT_SIZE_NORM = 100


def create_test_vectors(
    size: int,
) -> tuple[list[float], list[float], Vector_float, Vector_float]:
    """Create Python lists and TNL vectors filled with random data."""
    a_list = [float(i * 1.7 + 0.3) for i in range(size)]
    b_list = [float(i * 0.9 + 1.1) for i in range(size)]

    a_tnl = Vector[float](size)
    b_tnl = Vector[float](size)
    for i in range(size):
        a_tnl.setElement(i, a_list[i])
        b_tnl.setElement(i, b_list[i])

    return a_list, b_list, a_tnl, b_tnl


def benchmark(label: str, func: Callable[[], None], runs: int = 3) -> float:
    """Run *func* multiple times, print the best time, and return it."""
    times: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    best = min(times)
    print(f"{label}: {best:.6f} seconds (best of {runs})")
    return best


def benchmark_add(size: int, runs: int) -> None:
    """Compare vector addition: Python loop vs TNL vs NumPy."""
    a, b, a_tnl, b_tnl = create_test_vectors(size)
    print(f"\n{'=' * 50}")
    print(f"Adding vectors of size {size}")

    # Python for-loop
    c_py: list[float] = [0.0] * size

    def python_add() -> None:
        for i in range(size):
            c_py[i] = a[i] + b[i]

    benchmark("Python for-loop", python_add, runs)

    # TNL vector addition
    def tnl_add() -> None:
        _ = a_tnl + b_tnl

    benchmark("TNL", tnl_add, runs)

    # NumPy addition
    a_np = np.array(a)
    b_np = np.array(b)

    def numpy_add() -> None:
        _ = np.add(a_np, b_np)

    benchmark("NumPy", numpy_add, runs)


def benchmark_dot(size: int, runs: int) -> None:
    """Compare dot product: Python loop vs TNL (via NumPy) vs NumPy."""
    a, b, a_tnl, b_tnl = create_test_vectors(size)
    print(f"\n{'=' * 50}")
    print(f"Dot product of vectors of size {size}")

    # Python for-loop
    def python_dot() -> None:
        _ = sum(a[i] * b[i] for i in range(size))

    benchmark("Python for-loop", python_dot, runs)

    # TNL via DLPack → NumPy vecdot
    # TODO: update once native TNL vector dot product is available
    a_tnl_np = np.from_dlpack(a_tnl)
    b_tnl_np = np.from_dlpack(b_tnl)

    def tnl_dot() -> None:
        np.vecdot(a_tnl_np, b_tnl_np)

    benchmark("TNL (via NumPy vecdot)", tnl_dot, runs)

    # NumPy vecdot
    a_np = np.array(a)
    b_np = np.array(b)

    def numpy_dot() -> None:
        np.vecdot(a_np, b_np)

    benchmark("NumPy", numpy_dot, runs)


def benchmark_norm(size: int, runs: int) -> None:
    """Compare vector norm: Python loop vs TNL (via NumPy) vs NumPy."""
    a, b, a_tnl, b_tnl = create_test_vectors(size)
    print(f"\n{'=' * 50}")
    print(f"Norm of two vectors of size {size}")

    # Python for-loop
    def python_norm() -> None:
        _ = math.sqrt(sum(x * x for x in a))
        _ = math.sqrt(sum(x * x for x in b))

    benchmark("Python for-loop", python_norm, runs)

    # TNL via DLPack → NumPy vector_norm
    # TODO: update once native TNL vector norm is available
    a_tnl_np = np.from_dlpack(a_tnl)
    b_tnl_np = np.from_dlpack(b_tnl)

    def tnl_norm() -> None:
        np.linalg.vector_norm(a_tnl_np)
        np.linalg.vector_norm(b_tnl_np)

    benchmark("TNL (via NumPy vector_norm)", tnl_norm, runs)

    # NumPy vector_norm
    a_np = np.array(a)
    b_np = np.array(b)

    def numpy_norm() -> None:
        np.linalg.vector_norm(a_np)
        np.linalg.vector_norm(b_np)

    benchmark("NumPy", numpy_norm, runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare vector operation speeds: Python vs PyTNL vs NumPy")
    parser.add_argument("--operation", choices=["add", "dot", "norm", "all"], default="all", help="Benchmark to run (default: all)")
    parser.add_argument("--size", type=int, default=None, help="Override vector size (default depends on operation)")
    parser.add_argument("--runs", type=int, default=3, help="Number of timing runs per benchmark (best of N)")
    args = parser.parse_args()

    op = args.operation
    size = args.size
    runs = args.runs

    if op in ("add", "all"):
        benchmark_add(size or DEFAULT_SIZE_ADD, runs)
    if op in ("dot", "all"):
        benchmark_dot(size or DEFAULT_SIZE_DOT, runs)
    if op in ("norm", "all"):
        benchmark_norm(size or DEFAULT_SIZE_NORM, runs)


if __name__ == "__main__":
    main()

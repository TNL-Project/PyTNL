# ruff: noqa: ANN001, ANN201
# mypy: disable-error-code="import-untyped,misc,no-untyped-def,no-untyped-usage,no-any-return,import-not-found"
# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownArgumentType=false, reportUntypedFunctionDecorator=false, reportUnknownMemberType=false, reportAssignmentType=false, reportCallIssue=false, reportIndexIssue=false, reportMissingImports=false
# needs additional dependencies: numba, numba-cuda, with uv you can run the example like this:
# uv run --with numba --with numba-cuda[cu13] examples/numba_jit.py
import numpy as np
from numba import cuda, jit, vectorize

from pytnl import devices
from pytnl.containers import Array, NDArray


@jit(nopython=True)
def fill_with_index_njit(data):
    """Fill a 1D array with values matching their indices."""
    for i in range(len(data)):
        data[i] = i


@vectorize(["float64(float64, float64)"], nopython=True)
def vectorized_scale(x, scale):
    """Element-wise scale implemented through Numba vectorize (CPU)."""
    return x * scale


@cuda.jit
def numba_scale_kernel_3d(data, scale, nx, ny, nz):
    """Scale a 3D NDArray element-wise."""
    i, j, k = cuda.grid(3)
    if i < nx and j < ny and k < nz:
        data[i, j, k] = data[i, j, k] * scale


# 1) Numba CPU functions run directly on PyTNL Array.
array = Array[float, devices.Host](8)
fill_with_index_njit(array)
print("Array after fill_with_index_njit(Array):", list(array))

# the out parameter is more strict and requires a numpy ArrayType, creating a view allows using the same memory
vectorized_scale(array, 2.0, out=np.asarray(array))
print("Array after vectorized_scale(Array, 2.0, out=Array):", list(array))


# 2) Numba CUDA kernel with PyTNL NDArray on device.
if cuda.is_available():
    nx, ny, nz = 40, 40, 40
    nd_cuda = NDArray[3, float, devices.Cuda]()
    nd_cuda.setSizes(nx, ny, nz)
    nd_cuda.setValue(3.0)

    threads = (4, 4, 4)
    blocks = (
        (nx + threads[0] - 1) // threads[0],
        (ny + threads[1] - 1) // threads[1],
        (nz + threads[2] - 1) // threads[2],
    )

    # uses DLPack from numba-cuda versions >= 0.29
    numba_scale_kernel_3d[blocks, threads](nd_cuda, 0.5, nx, ny, nz)

    print("CUDA NDArray sample [0, 0, 0] after kernel:", nd_cuda[0, 0, 0])


# 3) Zero-copy NumPy views through Buffer protocol
array_np = np.asarray(array)
print("NumPy shares memory with Array:", np.shares_memory(array_np, np.from_dlpack(array)))

# or through DLPack
nd = NDArray[3, float, devices.Host]()
nd.setSizes(2, 3, 4)
nd_np = np.from_dlpack(nd)
nd_np[:] = 1.5

print("Host NDArray shape:", nd_np.shape)
print("Host NDArray sample [1, 2, 3]:", nd[1, 2, 3])

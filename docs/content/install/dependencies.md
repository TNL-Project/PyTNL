- **Python 3.12 or later**, including the _development headers_ for building
  C/C++ Python modules
- **Compiler for the C++17 standard**, e.g. [GCC](https://gcc.gnu.org/) or [Clang](https://clang.llvm.org/)
- **[Git](https://git-scm.com/)**
- **An MPI library** such as [OpenMPI](https://www.open-mpi.org/)
- _(Optional):_ [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) for building and using CUDA-enabled PyTNL
  submodules

You can install all dependencies with one of the following commands, depending
on your Linux distribution:

### Arch Linux

```shell
pacman -S base-devel git python openmpi
```

### Ubuntu

```shell
apt install build-essential git python3-dev libopenmpi-dev
```

Additional dependencies will be pulled in automatically either as Python
packages (e.g. [cmake](https://pypi.org/project/cmake/)) or using the [FetchContent cmake module](https://cmake.org/cmake/help/latest/module/FetchContent.html).

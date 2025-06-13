# PyTNL

Python bindings for the Template Numerical Library (TNL).

> **Note:** pyproject.toml requires **Python 3.9 or later**.

## Installation

To install PyTNL from source, first make sure that [Git](https://git-scm.com/)
is installed and clone the repository:

    git clone https://gitlab.com/tnl-project/pytnl.git

### Using pip

PyTNL can be installed as a standard Python package using
[pip](https://pip.pypa.io/en/latest/getting-started/) as follows:

    python -m pip install path/to/pytnl

where `path/to/pytnl` is the directory where you cloned the PyTNL repository.

This will install [CMake from PyPI](https://pypi.org/project/cmake/) and start
the compilation from C++ source code. The following dependencies are needed for
the installation to succeed: a C++ compiler (e.g. [GCC](https://gcc.gnu.org/)
or [Clang](https://clang.llvm.org/)) and an [MPI](https://www.mpi-forum.org/).

You can install all dependencies with one of the following commands, depending
on your Linux distribution:

- Arch Linux:

      pacman -S base-devel git python openmpi

- Ubuntu:

      apt install build-essential git python3-dev libopenmpi-dev

### Using CMake

To install PyTNL using _CMake_, the following dependencies are needed:
[CMake](https://cmake.org/), [GNU Make](https://www.gnu.org/software/make/),
a C++ compiler (e.g. [GCC](https://gcc.gnu.org/) or [Clang](https://clang.llvm.org/)),
[Python 3](https://www.python.org/), and an [MPI](https://www.mpi-forum.org/).

You can install all dependencies with one of the following commands, depending
on your Linux distribution:

- Arch Linux:

      pacman -S base-devel git cmake python openmpi

- Ubuntu:

      apt install build-essential git cmake python3-dev libopenmpi-dev

Build and install PyTNL:

    cd pytnl
    cmake -B build -S . -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
    cmake --build build
    cmake --install build

## Usage

After installing PyTNL, run `python` and import some module from the `pytnl`
package, e.g. `pytnl.containers`.

The [examples directory](./examples/) contains some short examples showing how
to use PyTNL.

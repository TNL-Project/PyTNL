# PyTNL

Python bindings for the Template Numerical Library

## Quickstart

1. Make sure that [Git](https://git-scm.com/), [CMake](https://cmake.org/),
   [GNU Make](https://www.gnu.org/software/make/), [Python 3](https://www.python.org/), and
   an [MPI](https://www.mpi-forum.org/) implementation are installed on your Linux system.

2. Clone the repository and initialize the submodules:

       git clone --recurse-submodules https://gitlab.com/tnl-project/pytnl.git

3. Build and install the Python bindings:

       cd pytnl
       cmake -B build -S . -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$HOME/.local"
       cmake --build build
       cmake --install build

4. Run `python` and import the `tnl` and/or `tnl_mpi` modules.

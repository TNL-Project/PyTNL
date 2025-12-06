This section covers the suggested setup for PyTNL developers.
First make sure to install all [dependencies](dependencies.md).

Clone the repository and create a Python **virtual environment** (venv) for the
project:

```shell
git clone https://gitlab.com/tnl-project/pytnl.git
cd pytnl
python -m venv .venv
source .venv/bin/activate
```

Next we need to install the build system in this environment:

```shell
pip install scikit-build-core
pip install cmake ninja  # only necessary if not present in your system
```

To facilitate repeatable builds, the following command installs PyTNL without
build isolation using the active venv and shared `build` subdirectory for build
artifacts:

```shell
pip install --no-build-isolation -ve .[dev]
```

Run the previous command again after making changes in the code to rebuild the
project.

The `[dev]` _extra_ also installs packages for testing and linting the code
that you can run:

```shell
pytest
ruff check
basedpyright
mypy
```

The `[dev-cuda]` _extra_ additionally contains dependencies necessary for
testing the CUDA support.

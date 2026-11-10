# Project Knowledge Base

**Generated:** 2026-11-10
**Commit:** 7ef9700
**Branch:** main

## Overview

Python bindings for the Template Numerical Library (TNL) — hybrid C++/Python project.
C++17 + CUDA via nanobind v3, built with CMake + scikit-build-core.
Requires Python >= 3.12 (PEP 695 `type` aliases are used in the wrappers) and MPI.

## Structure

```
pytnl/
├── src/pytnl/          # Python package (thin .py wrappers + C++ binding sources)
│   ├── _meta.py        # CPPClassTemplate metaclass, type aliases (VT, DT, DIMS)
│   ├── devices.py      # Host/Cuda device selector classes
│   ├── containers/     # Array, ArrayView, Vector, VectorView, StaticVector,
│   │                   #   NDArray, NDArrayView, NDArrayIndexer, DistributedNDArray
│   ├── matrices/       # SparseMatrix/DenseMatrix + views + row views, formats,
│   │                   #   ElementsOrganization, copySparseMatrix
│   ├── solvers/        # ODESolver[Method, Device] — 18 ODE methods (Host + CUDA),
│   │                   #   IterativeSolver, ExplicitSolver
│   └── meshes/         # Grid, Mesh, DistributedMesh, VTK/VTU/PVTU readers/writers
├── include/pytnl/      # C++ headers for the binding layer (not TNL itself)
├── tests/              # pytest suite — CPU + CUDA variants, hypothesis-based
├── examples/           # Standalone usage scripts
├── docs/               # MkDocs documentation (material theme)
├── CMakeLists.txt      # Root CMake: FetchContent TNL+nanobind+tinyxml2, MPI, ZLIB
├── pyproject.toml      # scikit-build-core, ruff/mypy/pyright/pytest config
└── justfile            # Task runner: install, check, test, format, release
```

Binding-layer C++ headers normally live in `include/pytnl/<module>/`, but matrix
binding headers (`MatrixBase.h`, `SparseMatrix.h`, `DenseMatrix.h`) live next to
the sources in `src/pytnl/matrices/`.

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| Add new C++ binding | `src/pytnl/<module>/` (.cpp/.cu) + `include/pytnl/<module>/` (.h) | Register in `src/pytnl/CMakeLists.txt` |
| Add Python wrapper class | `src/pytnl/<module>/__init__.py` | Must use `CPPClassTemplate` metaclass from `_meta.py` |
| Add CUDA variant | Create `.cu` file alongside `.cpp`, add `_cuda` module in CMakeLists.txt | Lazy-imported at runtime via `CPPClassTemplate._device_parameter` |
| Add ODE method | `src/pytnl/solvers/ode_methods.cpp` + overloads/`__all__` in `solvers/__init__.py` | Exported into the `ode_methods` submodule; solver classes use `ODESolver_<Method>` naming |
| Add sparse/dense matrix format or value type | `src/pytnl/matrices/{matrices.cpp,MatrixBase.h,SparseMatrix.h,DenseMatrix.h}` + `matrices/__init__.py` | Format tags are empty classes in the `formats` submodule; add overloads and class-name mapping |
| Add test | `tests/test_<Feature>.py` + `tests/test_<Feature>_cuda.py` if GPU | Use `@pytest.mark.cuda` / `@pytest.mark.mpi` markers |
| Test environment setup | `tests/conftest.py` | Sets `OMPI_MCA_accelerator` from markers and `OMP_NUM_THREADS=4` |
| Type checking fixes | `src/pytnl/__init__.py` is empty — stubs live in build dir (`*.pyi`) | `reportMissingModuleSource = "none"` in pyright |
| CI pipeline | `.gitlab-ci.yml` | GitLab-native, not GitHub Actions |
| Build commands | `justfile` | `just install`, `just check`, `just test`, `just format` |

## Code map

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `CPPClassTemplate` | Metaclass | `src/pytnl/_meta.py:32` | Resolves `Class[Args]` syntax to C++ nanobind classes |
| `VT` | TypeAlias | `src/pytnl/_meta.py:10` | `int \| float \| complex` |
| `DT` | TypeAlias | `src/pytnl/_meta.py:13` | `Host \| Cuda` |
| `DIMS` | TypeAlias | `src/pytnl/_meta.py:16` | `Literal[1, 2, 3]` |
| `is_dim_guard` | Function | `src/pytnl/_meta.py:22` | Runtime `TypeGuard` for `DIMS` |
| `AbstractDevice` | Class | `src/pytnl/devices.py:1` | Base for device type hierarchy |
| `Host` | Class | `src/pytnl/devices.py:8` | CPU device selector |
| `Cuda` | Class | `src/pytnl/devices.py:18` | GPU device selector |
| `Array` | Template wrapper | `src/pytnl/containers/__init__.py:111` | `Array[ValueType, DeviceType]` |
| `ArrayView` | Template wrapper | `src/pytnl/containers/__init__.py:211` | `ArrayView[ValueType, DeviceType]` |
| `Vector` | Template wrapper | `src/pytnl/containers/__init__.py:297` | `Vector[ValueType, DeviceType]` |
| `VectorView` | Template wrapper | `src/pytnl/containers/__init__.py:481` | `VectorView[ValueType, DeviceType]` |
| `StaticVector` | Template wrapper | `src/pytnl/containers/__init__.py:398` | `StaticVector[Dim, ValueType]` |
| `NDArray` | Template wrapper | `src/pytnl/containers/__init__.py:650` | `NDArray[Dim, ValueType, DeviceType]` |
| `NDArrayView` | Template wrapper | `src/pytnl/containers/__init__.py:819` | `NDArrayView[Dim, ValueType, DeviceType]` |
| `NDArrayIndexer` | Template wrapper | `src/pytnl/containers/__init__.py:871` | `NDArrayIndexer[Dim]` — index/strides/overlaps helper |
| `DistributedNDArray` | Template wrapper | `src/pytnl/containers/__init__.py:1037` | `DistributedNDArray[NDArrayType]` (same-module dispatch) |
| `Grid` | Template wrapper | `src/pytnl/meshes/__init__.py:133` | `Grid[Dim, DeviceType]` |
| `Mesh` | Template wrapper | `src/pytnl/meshes/__init__.py:269` | `Mesh[Topology, DeviceType]` |
| `DistributedMesh` | Template wrapper | `src/pytnl/meshes/__init__.py:413` | `DistributedMesh[MeshType]` (same-module dispatch) |
| `VTIWriter` / `VTUWriter` / `VTKWriter` / `PVTUWriter` | Template wrappers | `src/pytnl/meshes/__init__.py:490`, `:624`, `:757`, `:842` | Mesh writers keyed by mesh/topology |
| `resolveMeshType`, `resolveAndLoadMesh` | Functions | `src/pytnl/meshes/__init__.py:855`, `:871` | Mesh type resolution and loading |
| `MeshReader` / `VTIReader` / `VTUReader` / `VTKReader` / `PVTUReader` / `getMeshReader` | Re-exports | `src/pytnl/meshes/__init__.py:12` | Direct imports from `pytnl._meshes` (no Pythonization) |
| `ODESolver` | Template wrapper | `src/pytnl/solvers/__init__.py:292` | `ODESolver[Method, DeviceType]` — 18 ODE methods |
| `ode_methods` | Submodule | `src/pytnl/solvers/__init__.py:8` | ODE method tag classes (`Euler`, `DormandPrince`, …) |
| `IterativeSolver` | Concrete class | `src/pytnl/solvers/__init__.py:309` | `IterativeSolver_float_int` from `_solvers` |
| `ExplicitSolver` | Concrete class | `src/pytnl/solvers/__init__.py:310` | `ExplicitSolver_float_int` from `_solvers` |
| `DenseMatrix` | Template wrapper | `src/pytnl/matrices/__init__.py:619` | `DenseMatrix[ValueType, DeviceType, Organization]` |
| `SparseMatrix` | Template wrapper | `src/pytnl/matrices/__init__.py:639` | `SparseMatrix[ValueType, DeviceType, Format]` |
| `DenseMatrixView` | Template wrapper | `src/pytnl/matrices/__init__.py:664` | Dense matrix view |
| `SparseMatrixView` | Template wrapper | `src/pytnl/matrices/__init__.py:684` | Sparse matrix view |
| `SparseMatrixRowView` / `SparseMatrixConstRowView` | Template wrappers | `src/pytnl/matrices/__init__.py:1093`, `:1109` | Sparse row views (shared by all formats) |
| `DenseMatrixRowView` / `DenseMatrixConstRowView` | Template wrappers | `src/pytnl/matrices/__init__.py:1120`, `:1132` | Dense row views |
| `formats`, `ElementsOrganization` | Submodule / Enum | `src/pytnl/matrices/__init__.py:8` (from `matrices.cpp:101`, `:128`) | CSR/Ellpack/SlicedEllpack tags and row/column-major enum |
| `copySparseMatrix` | Function | `src/pytnl/matrices/__init__.py:1143` | Format/device copy dispatch (Host ↔ CUDA) |

## Conventions

- **Commit messages**: Conventional Commits (`type(scope): description`), e.g. `fix(examples): guard against zero dimensions`.
  Types: `fix`, `feat`, `refactor`, `perf`, `docs`, `test`, `ci`, `chore`, `build`, `style`.
  Body formatting: ≤72 chars per line, use markdown where appropriate.
- **Assisted-by trailer**: When AI tools contribute, add `Assisted-by: AGENT_NAME:MODEL_VERSION` (Linux kernel convention from `Documentation/process/coding-assistants.rst`).
  Example: `Assisted-by: Opencode:glm-5.1`.
  Optional tool names may follow: `Assisted-by: Claude:claude-3-opus coccinelle sparse`.
  Do NOT use `Signed-off-by` for AI — only humans certify DCO.
- **C++ indent: 3 spaces**, column limit 128 (`.clang-format`)
- **Python indent: 4 spaces**, line length 160 (ruff)
- **CMake formatting**: gersemi with `.gersemirc` (indent 4, line length 128, `list_expansion: favour-inlining`)
- **Comment style**: Section comments use plain text without decorative separators — write `// Base classes`, not `// ---- Base classes ----`. Never add `----` or similar divider lines in comments.
- **Template wrapper pattern**: Python class with `CPPClassTemplate` metaclass → `Class[args]` resolves to C++ nanobind class via naming convention (`Class_Arg1_Arg2`).
  Wrappers provide hand-written `@overload` chains for type narrowing; the metaclass is only the runtime fallback.
- **Device dispatch**: `_device_parameter` appends `_<device-name-lowercase>` to the module name (`Host` is special-cased to no suffix, because it is the base module).
  `_dispatch_same_module_parameter` resolves the module from the template argument's own `__module__`.
- **Device default**: `Host` is always the default device when omitted from subscript.
- **Overload ordering**: Overloads must be ordered most-specific first following Python's numeric tower: `int`, `float`, `complex` (since `float` accepts `int` and `complex` accepts `float`).
  Broader types must come last to avoid being shadowed.
- **CUDA modules**: Lazy-imported at runtime, `TYPE_CHECKING`-only imports for type checkers.
- **Type ignore conventions**: Minimize ignore comments in general.
  `# type: ignore` and `# pyright: ignore` must NOT be on the same line — pyright ignores all diagnostics when it sees `# type: ignore`, making a trailing `# pyright: ignore` a no-op.
  Use `# type: ignore[code]` for mypy and `# pyright: ignore[rule]` for pyright independently.
  If both mypy and pyright suppression is needed, put them on separate lines or use `# type: ignore`.
- **Typing for CUDA modules**: CUDA is optional, but type checkers must pass even when it is not available.
  Use `# type: ignore[import-not-found, unused-ignore]` to make it pass both with and without CUDA.
  In CUDA test files (`tests/test_*_cuda.py`), use a file-level `# mypy: disable-error-code="..."` directive for frequent error codes
  (those that cascade across many lines, e.g. `attr-defined`, `no-any-unimported`, `unused-ignore`, `valid-type`)
  and inline `# type: ignore[code]` for rare, isolated occurrences (e.g. `operator`, `misc`, `no-any-return`).
  basedmypy only honors the first `# mypy: disable-error-code=` line per file — keep all codes on a single line.
- **CPPClassTemplate in annotations**: `Array[float]` works at runtime but triggers `reportInvalidTypeArguments` in type annotations. Omit explicit annotations (`a = Array[float]()` — correctly inferred as `Array_float`) or use concrete types from `_containers` when annotation is needed (e.g. `-> Vector_float`). See [gitlab#33](https://gitlab.com/tnl-project/pytnl/-/work_items/33).
- **Stub generation**: Custom `add_custom_command` invokes nanobind's `stubgen.py` directly (not `nanobind_add_stub()`).
  Key flags: `-S` (bypass `site.py`, so the editable-install `.pth` finder cannot redirect imports to stale `.so` files), fully-qualified `-m pytnl.<target>` (nanobind >= 2.15 writes absolute self-imports), `-O` output directory (recursive mode may emit multiple `.pyi` files for `def_submodule`s), `PYTHONPATH` override, and `OMPI_MCA_accelerator=null`.
  Module-to-stub mapping lives in `src/pytnl/CMakeLists.txt` (e.g. `_matrices` → `_matrices/__init__.pyi` + `_matrices/formats.pyi`).
- **`NB_SUPPRESS_WARNINGS`**: Passed to every `nanobind_add_module` call. It adds nanobind/Python includes via `-isystem` and disables compiler + clang-tidy diagnostics for nanobind headers (needed because nanobind targets are created lazily, so `FetchContent_Declare(... SYSTEM)` cannot act on them).
- **clang-tidy**: Configured in `.clang-tidy` (readability/modernize/performance/mpi checks; several readability checks disabled). `just check-clang-tidy` only runs on a **Debug build with `PyTNL_USE_CUDA=OFF`** and requires `build/compile_commands.json`; otherwise the recipe prints a skip message and exits 0.
- **Complex value types**: Matrices support `float` (C++ `double`) and `complex`. Host uses `std::complex<double>`, CUDA uses `TNL::Arithmetics::Complex<double>` — distinct C++ types, so cross-device `copySparseMatrix` is **not** available for complex matrices.
- **Sparse format tags**: `SparseMatrix` takes `formats.CSR` (default), `formats.Ellpack`, or `formats.SlicedEllpack` as a third subscript argument. The two-argument form is always `(value_type, device_type)`. Format tags are empty C++ classes exported into the `formats` submodule.
- **Build isolation broken**: `build-dir = "build"` enables incremental builds but breaks PEP 517 isolation.
- **Venv-managed tooling**: justfile uses `.venv/bin/` for Python tools (pytest, basedpyright, mypy); system ruff for linting/formatting; `_ensure-venv` auto-creates venv.
- **`from __future__ import annotations`**: Do NOT add to new files in examples/ or tests/. The import is deprecated in Python 3.14 (PEP 649/749 deferred evaluation replaces it). Kept in `src/pytnl/` only because `_containers_cuda`/`_meshes_cuda`/`_matrices_cuda`/`_solvers_cuda` are `TYPE_CHECKING`-only imports used in annotations.
- **No `__main__.py`**: Library-only package, no CLI entry points.
- **Empty root `__init__.py`**: Submodules are the public API surface.

## Anti-patterns (this project)

- DO NOT add `[project.scripts]` — this is a library, not a CLI tool
- DO NOT use `find -exec` in justfile — use `xargs` instead (preserves exit code)
- DO NOT sort C++ includes — `.clang-format` has `SortIncludes: Never` due to cyclic inclusion
- DO NOT enable IPO/LTO on Alpine — inlining fails with `vsnprintf`
- DO NOT run CUDA tests without `-m cuda -n 0` flag combination
- DO NOT import CUDA modules unconditionally — use lazy import or `TYPE_CHECKING` guard
- DO NOT use `nanobind_add_stub()` — it breaks on the submodule layout; use the custom stubgen command
- DO NOT run `check-clang-tidy` against a CUDA or non-Debug build — reconfigure with `-DPyTNL_USE_CUDA=OFF -DCMAKE_BUILD_TYPE=Debug`
- DO NOT put `# type: ignore` and `# pyright: ignore` on the same line (the pyright one becomes a no-op)
- DO NOT copy complex matrices across devices (incompatible C++ complex types)
- DO NOT use `Signed-off-by` for AI contributions — use `Assisted-by`

## Unique styles

- `CPPClassTemplate` metaclass system: Python `__getitem__` on the metaclass simulates C++ template instantiation
- Dual device module architecture: `pytnl._<module>` (Host) vs `pytnl._<module>_cuda` (Cuda) for containers, matrices, meshes, and solvers
- Extensive `@overload` decorators for precise type narrowing on template subscript results
- `basedmypy` + `basedpyright` dual type checking (non-standard mypy/pyright forks)
- `just` task runner instead of Makefile
- gersemi for CMake formatting (`.gersemirc`)
- clang-tidy for C++ linting (`.clang-tidy`), shellcheck for inline justfile recipes
- GitLab CI (not GitHub Actions)

## Commands

```bash
# Setup (creates .venv, installs project + dev deps)
just install                        # with CUDA support (default), installs [dev,cuda]
just install host                   # CPU-only build (PyTNL_USE_CUDA=OFF, skips CUDA modules)

# All checks (format + ruff + clang-tidy + typing + typos + shellcheck)
just check

# Individual checks
just check-code                     # ruff check (system ruff)
just check-format                   # just --fmt --check + clang-format --dry-run + gersemi --diff + ruff format --diff
just check-clang-tidy               # run-clang-tidy (Debug build, PyTNL_USE_CUDA=OFF required)
just check-typing                   # basedpyright + mypy (venv)
just check-typos                    # typos --sort
just check-recipes                  # shellcheck on inline bash in justfile recipes

# Format
just format                         # just --fmt + clang-format -i + gersemi --in-place + ruff format

# Build
just build                          # sdist + wheel (--no-isolation)
just build-sdist                    # sdist only
just build-wheel                    # wheel only

# Test (extra args are forwarded to pytest)
just test                           # pytest (excludes CUDA by default, xdist auto)
just test -m cuda -n 0              # CUDA tests (GPU required)
mpirun -np 2 just test -m mpi -n 0  # MPI tests (run through mpirun)

# Docs
mkdocs serve                        # local docs preview
mkdocs build                        # static site into public/

# Clean
just clean                          # rm -frv dist build
```

## Notes

- Python >= 3.12 required; CI covers 3.12, 3.13, and 3.14 on Alpine and Debian
- numpy>=2.2.5 required (older versions break `np.from_dlpack()` writable flag)
- `mpi4py` is a core dependency (not optional); MPI C++ library is required at build time
- Bundled dependency versions: TNL 0.4.3, nanobind v3.1.0, tinyxml2 11.0.0
- CMake options: `PyTNL_USE_CUDA` (default ON), `PyTNL_ENABLE_INTERPROCEDURAL_OPTIMIZATION` (default ON), `PyTNL_EXPORT_INTERFACE_TARGETS` (default when top-level)
- CUDA builds default to `CMAKE_CUDA_ARCHITECTURES=native`
- `distributeFaces` is host-only (no CUDA variant)
- `distributeSubentities` is host-only (no CUDA variant)
- Only 1D/2D/3D have bindings — higher dimensions would need C++ additions
- `PVTIReader` not yet implemented (`.pvti` cases are TODO in mesh tests)
- Slicing works for `Array` and `Vector` (Host and CUDA) but is not implemented for `NDArray`, views (`ArrayView`, `VectorView`, `NDArrayView`), or `StaticArray`
- CuPy `writable` flag issue tracked upstream (cupy/cupy#2616)
- `numba<0.66` is pinned in the `cuda` extra

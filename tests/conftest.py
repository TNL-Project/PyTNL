import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Avoid Open MPI abort on dual-accelerator hosts or runners without GPU hardware.
    # When cuda-marked tests are selected, tell Open MPI to use the CUDA accelerator;
    # otherwise disable accelerator detection entirely.
    # Users can override via the OMPI_MCA_accelerator environment variable.
    if "OMPI_MCA_accelerator" not in os.environ:
        markexpr = str(getattr(config.option, "markexpr", ""))
        # Heuristic: if "cuda" appears positively in the mark expression, assume CUDA
        os.environ["OMPI_MCA_accelerator"] = "cuda" if "cuda" in markexpr and "not cuda" not in markexpr else "null"

    # Limit number of OpenMP threads used by TNL host backend.
    # Tests run in parallel and letting OpenMP use the maximum available CPU cores
    # causes oversubscription and test failures due to hypothesis deadline.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "4"

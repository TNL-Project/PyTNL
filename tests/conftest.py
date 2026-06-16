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

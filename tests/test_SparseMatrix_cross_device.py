# mypy: disable-error-code="attr-defined, no-any-unimported, unused-ignore, valid-type"
# pyright: standard
# pyright: reportMissingImports=information
# pyright: reportInvalidTypeForm=none

from typing import TYPE_CHECKING, cast

import pytest

import pytnl._matrices
from pytnl._containers import Vector_int
from pytnl.matrices import copySparseMatrix

if TYPE_CHECKING:
    import pytnl._containers_cuda as _containers_cuda  # type: ignore[import-not-found]
    import pytnl._matrices_cuda as _matrices_cuda  # type: ignore[import-not-found]
else:
    _containers_cuda = pytest.importorskip("pytnl._containers_cuda")
    _matrices_cuda = pytest.importorskip("pytnl._matrices_cuda")

pytestmark = pytest.mark.cuda

H_CSR = pytnl._matrices.SparseMatrix_float_CSR
H_Ellpack = pytnl._matrices.SparseMatrix_float_Ellpack
H_SlicedEllpack = pytnl._matrices.SparseMatrix_float_SlicedEllpack

C_CSR = _matrices_cuda.SparseMatrix_float_CSR
C_Ellpack = _matrices_cuda.SparseMatrix_float_Ellpack
C_SlicedEllpack = _matrices_cuda.SparseMatrix_float_SlicedEllpack

HostMatrix = H_CSR | H_Ellpack | H_SlicedEllpack
CudaMatrix = C_CSR | C_Ellpack | C_SlicedEllpack
AnySparseMatrix = HostMatrix | CudaMatrix

HOST_FORMATS: list[tuple[str, type[HostMatrix]]] = [
    ("CSR", H_CSR),
    ("Ellpack", H_Ellpack),
    ("SlicedEllpack", H_SlicedEllpack),
]

CUDA_FORMATS: list[tuple[str, type[CudaMatrix]]] = [
    ("CSR", C_CSR),
    ("Ellpack", C_Ellpack),
    ("SlicedEllpack", C_SlicedEllpack),
]

ENTRIES: list[tuple[int, int, float]] = [
    (0, 0, 1.0),
    (0, 2, 2.5),
    (1, 1, -3.0),
    (2, 0, 4.0),
    (2, 4, 7.0),
    (3, 3, 0.5),
    (4, 1, 9.0),
]

ROWS = 5
COLS = 5


def create_host_matrix(matrix_type: type[HostMatrix], entries: list[tuple[int, int, float]], rows: int, cols: int) -> HostMatrix:
    m = matrix_type()
    m.setDimensions(rows, cols)
    caps = Vector_int(rows)
    for i in range(rows):
        caps[i] = 0
    for r, _c, _v in entries:
        caps[r] += 1
    m.setRowCapacities(caps)
    for r, c, v in entries:
        m.setElement(r, c, v)
    return m


def create_cuda_matrix(matrix_type: type[CudaMatrix], entries: list[tuple[int, int, float]], rows: int, cols: int) -> CudaMatrix:
    m = matrix_type()  # type: ignore[misc]
    m.setDimensions(rows, cols)
    caps = _containers_cuda.Vector_int(rows)
    for i in range(rows):
        caps[i] = 0
    for r, _c, _v in entries:
        caps[r] += 1
    m.setRowCapacities(caps)
    for r, c, v in entries:
        m.setElement(r, c, v)
    return cast(CudaMatrix, m)


def assert_equal_elements(m: AnySparseMatrix, entries: list[tuple[int, int, float]], rows: int, cols: int) -> None:
    for r, c, v in entries:
        assert m.getElement(r, c) == pytest.approx(v)
    assert m.getElement(0, 1) == 0.0
    assert m.getElement(1, 0) == 0.0
    assert m.getElement(4, 4) == 0.0
    assert m.getRows() == rows
    assert m.getColumns() == cols


# ---------------------------------------------------------------------------
# Host → Cuda (all 9 format pairs)
# ---------------------------------------------------------------------------

_HOST_TO_CUDA_PAIRS: list[tuple[str, type[HostMatrix], str, type[CudaMatrix]]] = [
    (h_name, h_type, c_name, c_type) for h_name, h_type in HOST_FORMATS for c_name, c_type in CUDA_FORMATS
]


@pytest.mark.parametrize(
    "h_name, h_type, c_name, c_type",
    _HOST_TO_CUDA_PAIRS,
    ids=[f"{h}->{c}" for h, _, c, _ in _HOST_TO_CUDA_PAIRS],
)
def test_copySparseMatrix_host_to_cuda(
    h_name: str,
    h_type: type[HostMatrix],
    c_name: str,
    c_type: type[CudaMatrix],
) -> None:
    """Host → Cuda copy preserves content across all 9 format pairs."""
    src = create_host_matrix(h_type, ENTRIES, ROWS, COLS)
    dst = c_type()  # type: ignore[misc]
    dst.setDimensions(ROWS, COLS)
    copySparseMatrix(dst, src)
    assert_equal_elements(dst, ENTRIES, ROWS, COLS)


# ---------------------------------------------------------------------------
# Cuda → Host (all 9 format pairs)
# ---------------------------------------------------------------------------

_CUDA_TO_HOST_PAIRS: list[tuple[str, type[CudaMatrix], str, type[HostMatrix]]] = [
    (c_name, c_type, h_name, h_type) for c_name, c_type in CUDA_FORMATS for h_name, h_type in HOST_FORMATS
]


@pytest.mark.parametrize(
    "c_name, c_type, h_name, h_type",
    _CUDA_TO_HOST_PAIRS,
    ids=[f"{c}->{h}" for c, _, h, _ in _CUDA_TO_HOST_PAIRS],
)
def test_copySparseMatrix_cuda_to_host(
    c_name: str,
    c_type: type[CudaMatrix],
    h_name: str,
    h_type: type[HostMatrix],
) -> None:
    """Cuda → Host copy preserves content across all 9 format pairs."""
    src = create_cuda_matrix(c_type, ENTRIES, ROWS, COLS)
    dst = h_type()
    dst.setDimensions(ROWS, COLS)
    copySparseMatrix(dst, src)
    assert_equal_elements(dst, ENTRIES, ROWS, COLS)


# ---------------------------------------------------------------------------
# Round-trip: Host → Cuda → Host (same format)
# ---------------------------------------------------------------------------

_ROUNDTRIP_PAIRS: list[tuple[str, type[HostMatrix], type[CudaMatrix]]] = [
    (name, h_type, c_type)
    for name, h_type, c_type in [
        ("CSR", H_CSR, C_CSR),
        ("Ellpack", H_Ellpack, C_Ellpack),
        ("SlicedEllpack", H_SlicedEllpack, C_SlicedEllpack),
    ]
]


@pytest.mark.parametrize(
    "name, h_type, c_type",
    _ROUNDTRIP_PAIRS,
    ids=[name for name, _, _ in _ROUNDTRIP_PAIRS],
)
def test_copySparseMatrix_host_cuda_roundtrip(
    name: str,
    h_type: type[HostMatrix],
    c_type: type[CudaMatrix],
) -> None:
    """Host → Cuda → Host round-trip preserves content for each format."""
    src = create_host_matrix(h_type, ENTRIES, ROWS, COLS)
    tmp = c_type()  # type: ignore[misc]
    tmp.setDimensions(ROWS, COLS)
    copySparseMatrix(tmp, src)

    dst = h_type()
    dst.setDimensions(ROWS, COLS)
    copySparseMatrix(dst, tmp)
    assert dst == src  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Round-trip with format change: Host → Cuda → Host (different format)
# ---------------------------------------------------------------------------

_ROUNDTRIP_CROSS_FORMAT: list[tuple[str, type[HostMatrix], str, type[CudaMatrix], str, type[HostMatrix]]] = [
    (src_name, src_type, mid_name, mid_type, dst_name, dst_type)
    for src_name, src_type in HOST_FORMATS
    for mid_name, mid_type in CUDA_FORMATS
    for dst_name, dst_type in HOST_FORMATS
]


@pytest.mark.parametrize(
    "src_name, src_type, mid_name, mid_type, dst_name, dst_type",
    _ROUNDTRIP_CROSS_FORMAT,
    ids=[f"{s}->{m}->{d}" for s, _, m, _, d, _ in _ROUNDTRIP_CROSS_FORMAT],
)
def test_copySparseMatrix_cross_device_cross_format_roundtrip(
    src_name: str,
    src_type: type[HostMatrix],
    mid_name: str,
    mid_type: type[CudaMatrix],
    dst_name: str,
    dst_type: type[HostMatrix],
) -> None:
    """Host → Cuda → Host round-trip preserves content even when the
    intermediate CUDA format differs from both source and destination.
    """
    src = create_host_matrix(src_type, ENTRIES, ROWS, COLS)
    mid = mid_type()  # type: ignore[misc]
    mid.setDimensions(ROWS, COLS)
    copySparseMatrix(mid, src)

    dst = dst_type()
    dst.setDimensions(ROWS, COLS)
    copySparseMatrix(dst, mid)
    assert_equal_elements(dst, ENTRIES, ROWS, COLS)

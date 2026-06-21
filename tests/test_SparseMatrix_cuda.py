# mypy: disable-error-code="attr-defined, no-any-unimported, unused-ignore, valid-type"
# pyright: standard
# pyright: reportMissingImports=information
# pyright: reportInvalidTypeForm=none

import os
import tempfile
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pytnl.containers import Vector
from pytnl.devices import Cuda
from pytnl.matrices import SparseMatrix, formats

if TYPE_CHECKING:
    import pytnl._containers_cuda as _containers_cuda  # type: ignore[import-not-found]
    import pytnl._matrices_cuda as _matrices_cuda  # type: ignore[import-not-found]
else:
    _containers_cuda = pytest.importorskip("pytnl._containers_cuda")
    _matrices_cuda = pytest.importorskip("pytnl._matrices_cuda")

copySparseMatrix = _matrices_cuda.copySparseMatrix

# Mark all tests in this module
pytestmark = pytest.mark.cuda

# ---------------------------------------------------------------------------
# Type variable constraining the three CUDA matrix formats
# ---------------------------------------------------------------------------

M = TypeVar(
    "M",
    _matrices_cuda.SparseMatrix_float_CSR,
    _matrices_cuda.SparseMatrix_float_Ellpack,
    _matrices_cuda.SparseMatrix_float_SlicedEllpack,
)
matrix_types: tuple[type, ...] = M.__constraints__

# Mapping from format name to C++ class for parametrization
CSR = _matrices_cuda.SparseMatrix_float_CSR
Ellpack = _matrices_cuda.SparseMatrix_float_Ellpack
SlicedEllpack = _matrices_cuda.SparseMatrix_float_SlicedEllpack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_matrix(
    matrix_type: type[CSR],
    rows: int,
    cols: int,
    entries: list[tuple[int, int, float]],
) -> CSR:
    """Create a sparse matrix and fill it with the given entries.

    ``entries`` is a list of ``(row, col, value)`` tuples.
    Row capacities must be set before calling ``setElement``.
    """
    m = matrix_type()  # type: ignore[misc]
    m.setDimensions(rows, cols)
    caps = _containers_cuda.Vector_int(rows)
    for i in range(rows):
        caps[i] = 0
    for r, c, v in entries:
        caps[r] += 1
    m.setRowCapacities(caps)
    for r, c, v in entries:
        m.setElement(r, c, v)
    return m  # type: ignore[no-any-return]


def identity_matrix(
    matrix_type: type[CSR],
    size: int,
) -> CSR:
    """Create a size x size identity matrix in the given format."""
    entries = [(i, i, 1.0) for i in range(size)]
    return create_matrix(matrix_type, size, size, entries)


# ---------------------------------------------------------------------------
# Subscript / pythonization: correct class resolution
# ---------------------------------------------------------------------------


def test_subscript_cuda() -> None:
    """Verify subscript syntax resolves to the correct CUDA C++ class."""
    assert SparseMatrix[float, Cuda] is CSR
    assert SparseMatrix[float, Cuda, formats.CSR] is CSR
    assert SparseMatrix[float, Cuda, formats.Ellpack] is Ellpack
    assert SparseMatrix[float, Cuda, formats.SlicedEllpack] is SlicedEllpack


def test_subscript_cuda_default_format() -> None:
    """SparseMatrix[float, Cuda] defaults to CSR format."""
    assert SparseMatrix[float, Cuda] is SparseMatrix[float, Cuda, formats.CSR]


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------


def test_construction_empty() -> None:
    """Default-constructed matrix has zero dimensions."""
    m = CSR()
    assert m.getRows() == 0
    assert m.getColumns() == 0


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_setDimensions(matrix_type: type[M]) -> None:
    """setDimensions must update getRows / getColumns."""
    m = matrix_type()
    m.setDimensions(3, 4)
    assert m.getRows() == 3
    assert m.getColumns() == 4


# ---------------------------------------------------------------------------
# Element access
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_setElement_getElement(matrix_type: type[M]) -> None:
    """setElement / getElement round-trip for all formats."""
    m = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)])  # type: ignore[arg-type]

    assert m.getElement(0, 0) == 1.0
    assert m.getElement(0, 2) == 2.5
    assert m.getElement(1, 1) == -3.0


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_addElement(matrix_type: type[M]) -> None:
    """addElement accumulates values at the same position."""
    m = create_matrix(matrix_type, 1, 2, [(0, 0, 1.0), (0, 1, 2.0)])  # type: ignore[arg-type]
    m.addElement(0, 0, 10.0, 1.0)
    m.addElement(0, 0, 20.0, 1.0)
    assert m.getElement(0, 0) == 31.0  # 1.0 + 10.0 + 20.0


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_reset(matrix_type: type[M]) -> None:
    """reset() must zero out dimensions."""
    m = create_matrix(matrix_type, 2, 2, [(0, 0, 1.0)])  # type: ignore[arg-type]
    assert m.getRows() == 2
    m.reset()
    assert m.getRows() == 0
    assert m.getColumns() == 0


# ---------------------------------------------------------------------------
# vectorProduct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_vectorProduct(matrix_type: type[M]) -> None:
    """vectorProduct of identity matrix x vector equals the input vector."""
    n = 3
    m = identity_matrix(matrix_type, n)  # type: ignore[arg-type]

    in_vec = Vector[float, Cuda](n)
    for i in range(n):
        in_vec[i] = float(i + 1)

    out_vec = Vector[float, Cuda](n)
    m.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)

    for i in range(n):
        assert out_vec[i] == pytest.approx(float(i + 1))


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_vectorProduct_with_factors(matrix_type: type[M]) -> None:
    """vectorProduct with matrixMult=2.0, outMult=3.0 on identity matrix."""
    n = 3
    m = identity_matrix(matrix_type, n)  # type: ignore[arg-type]

    in_vec = Vector[float, Cuda](n)
    for i in range(n):
        in_vec[i] = float(i + 1)

    out_vec = Vector[float, Cuda](n)
    for i in range(n):
        out_vec[i] = 10.0  # pre-fill

    # out = 2.0 * (I * in) + 3.0 * out
    m.vectorProduct(in_vec, out_vec, 2.0, 3.0, 0, 0)

    for i in range(n):
        expected = 2.0 * float(i + 1) + 3.0 * 10.0
        assert out_vec[i] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_equality(matrix_type: type[M]) -> None:
    """Two matrices with identical content are equal by ==."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    m1 = create_matrix(matrix_type, 2, 3, entries)  # type: ignore[arg-type]
    m2 = create_matrix(matrix_type, 2, 3, entries)  # type: ignore[arg-type]
    assert m1 == m2  # type: ignore[operator]


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_inequality(matrix_type: type[M]) -> None:
    """Matrices with different content must not be equal."""
    m1 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (0, 2, 2.5)])  # type: ignore[arg-type]
    m2 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (0, 2, 9.9)])  # type: ignore[arg-type]
    assert m1 != m2  # type: ignore[operator]

    # Different number of non-zero elements
    m3 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0)])  # type: ignore[arg-type]
    m4 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (1, 1, 2.0)])  # type: ignore[arg-type]
    assert m3 != m4  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_getSerializationType() -> None:
    """getSerializationType returns a non-empty string; different formats differ."""
    csr_type = CSR.getSerializationType()
    ell_type = Ellpack.getSerializationType()
    se_type = SlicedEllpack.getSerializationType()

    assert isinstance(csr_type, str)
    assert len(csr_type) > 0
    assert isinstance(ell_type, str)
    assert len(ell_type) > 0
    assert isinstance(se_type, str)
    assert len(se_type) > 0

    # At least CSR should differ from non-CSR formats
    assert csr_type != ell_type or csr_type != se_type


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_save_load_roundtrip(matrix_type: type[M]) -> None:
    """save / load round-trip must preserve content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    m1 = create_matrix(matrix_type, 2, 3, entries)  # type: ignore[arg-type]

    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmpfile:
        filename = tmpfile.name

    try:
        m1.save(str(filename))
        m2 = matrix_type()
        m2.load(str(filename))
        assert m2.getRows() == m1.getRows()
        assert m2.getColumns() == m1.getColumns()
        assert m2 == m1
    finally:
        os.unlink(filename)


# ---------------------------------------------------------------------------
# Row capacities and compressed lengths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_row_capacities(matrix_type: type[M]) -> None:
    """setRowCapacities / getRowCapacities / getCompressedRowLengths are callable."""
    entries = [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (2, 0, 4.0), (2, 1, 5.0), (2, 2, 6.0)]
    m = create_matrix(matrix_type, 3, 4, entries)  # type: ignore[arg-type]

    # getRowCapacities writes into a pre-allocated vector
    result = _containers_cuda.Vector_int(3)
    m.getRowCapacities(result)
    assert result.getSize() == 3
    # Row capacities should be >= the number of non-zero elements per row
    assert result[0] >= 2
    assert result[1] >= 1
    assert result[2] >= 3

    # getRowCapacity for individual rows
    row_cap = m.getRowCapacity(0)
    assert row_cap >= 2

    # getCompressedRowLengths returns actual non-zero counts per row
    cl = _containers_cuda.Vector_int(3)
    m.getCompressedRowLengths(cl)
    assert cl.getSize() == 3
    assert cl[0] == 2
    assert cl[1] == 1
    assert cl[2] == 3


# ---------------------------------------------------------------------------
# Nonzero and allocation counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_nonzero_counts(matrix_type: type[M]) -> None:
    """After filling a matrix, getNonzeroElementsCount must match entry count."""
    entries = [(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 1, 5.0)]
    m = create_matrix(matrix_type, 3, 3, entries)  # type: ignore[arg-type]

    assert m.getNonzeroElementsCount() == 5
    assert m.getAllocatedElementsCount() >= m.getNonzeroElementsCount()


# ---------------------------------------------------------------------------
# Internal accessors (getValues, getColumnIndexes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_getValues_getColumnIndexes(matrix_type: type[M]) -> None:
    """Internal accessors return valid objects."""
    entries = [(0, 0, 1.0), (1, 1, 2.0)]
    m = create_matrix(matrix_type, 2, 2, entries)  # type: ignore[arg-type]

    values = m.getValues()
    assert values is not None
    assert values.getSize() > 0

    col_idx = m.getColumnIndexes()
    assert col_idx is not None
    assert col_idx.getSize() > 0


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_segments_methods(matrix_type: type[M]) -> None:
    """getAllocatedElementsCount and getNonzeroElementsCount return sensible values."""
    entries = [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)]
    m = create_matrix(matrix_type, 2, 2, entries)  # type: ignore[arg-type]

    assert m.getRows() == 2
    assert m.getColumns() == 2
    assert m.getNonzeroElementsCount() == 3
    assert m.getAllocatedElementsCount() >= 3


# ---------------------------------------------------------------------------
# copySparseMatrix — all 6 conversion pairs
# ---------------------------------------------------------------------------


def test_copySparseMatrix_csr_to_ellpack_and_back() -> None:
    """CSR → Ellpack → CSR round-trip preserves content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 0, 4.0)]
    source = create_matrix(CSR, 3, 3, entries)

    dest_ell = Ellpack()
    dest_ell.setDimensions(3, 3)
    copySparseMatrix(source, dest_ell)

    dest_csr = CSR()
    dest_csr.setDimensions(3, 3)
    copySparseMatrix(dest_ell, dest_csr)
    assert dest_csr == source


def test_copySparseMatrix_csr_to_sliced_ellpack_and_back() -> None:
    """CSR → SlicedEllpack → CSR round-trip preserves content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 0, 4.0)]
    source = create_matrix(CSR, 3, 3, entries)

    dest_se = SlicedEllpack()
    dest_se.setDimensions(3, 3)
    copySparseMatrix(source, dest_se)

    dest_csr = CSR()
    dest_csr.setDimensions(3, 3)
    copySparseMatrix(dest_se, dest_csr)
    assert dest_csr == source


def test_copySparseMatrix_ellpack_to_sliced_ellpack_and_back() -> None:
    """Ellpack → SlicedEllpack → Ellpack round-trip preserves content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 0, 4.0)]
    source = create_matrix(Ellpack, 3, 3, entries)  # type: ignore[arg-type]

    dest_se = SlicedEllpack()
    dest_se.setDimensions(3, 3)
    copySparseMatrix(source, dest_se)

    dest_ell = Ellpack()
    dest_ell.setDimensions(3, 3)
    copySparseMatrix(dest_se, dest_ell)
    assert dest_ell == source


def test_copySparseMatrix_all_pairs() -> None:
    """All six conversion pairs must not raise and produce equal content.

    Cross-format comparison (e.g. CSR == Ellpack) returns NotImplemented
    in nanobind, so we verify by converting back to the original format.
    """
    entries = [(0, 0, 1.0), (1, 1, 2.0)]
    source_csr = create_matrix(CSR, 2, 2, entries)
    source_ell = create_matrix(Ellpack, 2, 2, entries)  # type: ignore[arg-type]
    source_se = create_matrix(SlicedEllpack, 2, 2, entries)  # type: ignore[arg-type]

    pairs: list[tuple[type, type, type]] = [
        (CSR, Ellpack, CSR),
        (Ellpack, CSR, Ellpack),
        (CSR, SlicedEllpack, CSR),
        (SlicedEllpack, CSR, SlicedEllpack),
        (Ellpack, SlicedEllpack, Ellpack),
        (SlicedEllpack, Ellpack, SlicedEllpack),
    ]

    for src_type, dst_type, roundtrip_type in pairs:
        if src_type is CSR:
            src = source_csr
        elif src_type is Ellpack:
            src = source_ell
        else:
            src = source_se

        dst = dst_type()
        dst.setDimensions(2, 2)
        copySparseMatrix(src, dst)

        # Round-trip back to original format to verify content equality
        roundtrip = roundtrip_type()
        roundtrip.setDimensions(2, 2)
        copySparseMatrix(dst, roundtrip)
        assert roundtrip == src


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_str_output(matrix_type: type[M]) -> None:
    """str(matrix) returns non-empty string containing row numbers."""
    m = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (1, 2, 3.0)])  # type: ignore[arg-type]
    s = str(m)
    assert isinstance(s, str)
    assert len(s) > 0
    # Should contain row indicators
    assert "0:" in s or "Row" in s or "[" in s


# ---------------------------------------------------------------------------
# assign
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_assign(matrix_type: type[M]) -> None:
    """assign copies content; result equals source."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    source = create_matrix(matrix_type, 2, 3, entries)  # type: ignore[arg-type]

    dest = matrix_type()
    dest.assign(source)  # type: ignore[arg-type]
    assert dest == source


# ---------------------------------------------------------------------------
# Hypothesis: property-based element round-trip
# ---------------------------------------------------------------------------


@st.composite
def sparse_entries_strategy(
    draw: st.DrawFn,
    rows: int,
    cols: int,
    max_elements: int,
) -> list[tuple[int, int, float]]:
    """Generate a list of (row, col, value) with unique coordinates."""
    coords = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=rows - 1),
                st.integers(min_value=0, max_value=cols - 1),
            ).filter(lambda p: p[0] < rows and p[1] < cols),
            min_size=0,
            max_size=max_elements,
            unique=True,
        )
    )
    values = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100),
            min_size=len(coords),
            max_size=len(coords),
        )
    )
    return [(r, c, v) for (r, c), v in zip(coords, values)]


@pytest.mark.parametrize("matrix_type", matrix_types)
@given(data=st.data())
def test_property_element_roundtrip(
    matrix_type: type[M],
    data: st.DataObject,
) -> None:
    """Property: set/get element round-trip correctly for random sparse patterns."""
    rows = data.draw(st.integers(min_value=2, max_value=6))
    cols = data.draw(st.integers(min_value=2, max_value=6))
    entries = data.draw(sparse_entries_strategy(rows, cols, max_elements=min(rows * cols, 10)))

    m = create_matrix(matrix_type, rows, cols, entries)  # type: ignore[arg-type]

    for r, c, expected in entries:
        assert m.getElement(r, c) == expected


@pytest.mark.parametrize("matrix_type", matrix_types)
@given(data=st.data())
@settings(deadline=None)
def test_property_vectorProduct(
    matrix_type: type[M],
    data: st.DataObject,
) -> None:
    """Property: sparse matrix-vector product matches numpy dense computation."""
    n = data.draw(st.integers(min_value=2, max_value=5))
    entries = data.draw(sparse_entries_strategy(n, n, max_elements=n * n))

    m = create_matrix(matrix_type, n, n, entries)  # type: ignore[arg-type]

    # Build dense numpy matrix for reference
    dense = np.zeros((n, n), dtype=np.float64)
    for r, c, v in entries:
        dense[r, c] += v

    # Random input vector
    in_values = data.draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10),
            min_size=n,
            max_size=n,
        )
    )

    in_vec = Vector[float, Cuda](n)
    for i, val in enumerate(in_values):
        in_vec[i] = val

    out_vec = Vector[float, Cuda](n)
    m.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)

    expected = dense @ np.array(in_values)
    for i in range(n):
        assert out_vec[i] == pytest.approx(expected[i], rel=1e-10)

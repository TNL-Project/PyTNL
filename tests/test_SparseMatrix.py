import os
import tempfile
from typing import TypeVar

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pytnl._matrices
from pytnl._containers import Vector_int
from pytnl.containers import Vector
from pytnl.devices import Host
from pytnl.matrices import SparseMatrix, SparseMatrixRowView, SparseMatrixView, copySparseMatrix, formats

# ---------------------------------------------------------------------------
# Type variable constraining the three host matrix formats
# ---------------------------------------------------------------------------

M = TypeVar(
    "M",
    pytnl._matrices.SparseMatrix_float_CSR,
    pytnl._matrices.SparseMatrix_float_Ellpack,
    pytnl._matrices.SparseMatrix_float_SlicedEllpack,
)
matrix_types: tuple[type, ...] = M.__constraints__

# Mapping from format name to C++ class for parametrization
CSR = pytnl._matrices.SparseMatrix_float_CSR
Ellpack = pytnl._matrices.SparseMatrix_float_Ellpack
SlicedEllpack = pytnl._matrices.SparseMatrix_float_SlicedEllpack

# Mapping from format name to mutable view class
SV_CSR = pytnl._matrices.SparseMatrixView_float_CSR
SV_Ellpack = pytnl._matrices.SparseMatrixView_float_Ellpack
SV_SlicedEllpack = pytnl._matrices.SparseMatrixView_float_SlicedEllpack

# Mapping from format name to const view class
SV_CSR_const = pytnl._matrices.SparseMatrixView_float_CSR_const
SV_Ellpack_const = pytnl._matrices.SparseMatrixView_float_Ellpack_const
SV_SlicedEllpack_const = pytnl._matrices.SparseMatrixView_float_SlicedEllpack_const


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_matrix(
    matrix_type: type[pytnl._matrices.SparseMatrix_float_CSR],
    rows: int,
    cols: int,
    entries: list[tuple[int, int, float]],
) -> pytnl._matrices.SparseMatrix_float_CSR:
    """Create a sparse matrix and fill it with the given entries.

    ``entries`` is a list of ``(row, col, value)`` tuples.
    Row capacities must be set before calling ``setElement``.
    """
    m = matrix_type()
    m.setDimensions(rows, cols)
    caps = Vector_int(rows)
    for i in range(rows):
        caps[i] = 0
    for r, c, v in entries:
        caps[r] += 1
    m.setRowCapacities(caps)
    for r, c, v in entries:
        m.setElement(r, c, v)
    return m


def identity_matrix(
    matrix_type: type[pytnl._matrices.SparseMatrix_float_CSR],
    size: int,
) -> pytnl._matrices.SparseMatrix_float_CSR:
    """Create a size x size identity matrix in the given format."""
    entries = [(i, i, 1.0) for i in range(size)]
    return create_matrix(matrix_type, size, size, entries)


# ---------------------------------------------------------------------------
# Subscript / pythonization: correct class resolution
# ---------------------------------------------------------------------------


def test_subscript_returns_correct_class() -> None:
    """Verify subscript syntax resolves to the correct C++ class."""
    assert SparseMatrix[float] is CSR
    assert SparseMatrix[float, Host] is CSR
    assert SparseMatrix[float, Host, formats.CSR] is CSR
    assert SparseMatrix[float, Host, formats.Ellpack] is Ellpack
    assert SparseMatrix[float, Host, formats.SlicedEllpack] is SlicedEllpack


def test_subscript_default_device() -> None:
    """SparseMatrix[float] defaults to Host device."""
    assert SparseMatrix[float] is SparseMatrix[float, Host]


def test_subscript_default_format() -> None:
    """SparseMatrix[float, Host] defaults to CSR format."""
    assert SparseMatrix[float, Host] is SparseMatrix[float, Host, formats.CSR]


def test_subscript_invalid_value_type() -> None:
    """Non-float value types must raise TypeError."""
    for bad_type in (int, complex, bool):
        with pytest.raises(TypeError):
            SparseMatrix[bad_type]  # type: ignore[index, unused-ignore]


def test_subscript_invalid_format() -> None:
    """Unsupported format types must raise an error."""

    class BadFormat:
        pass

    for bad_fmt in (BadFormat,):
        with pytest.raises(ValueError):
            SparseMatrix[float, Host, bad_fmt]  # type: ignore[index]


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

    view = m.getView()
    cview = m.getConstView()

    # Test on owning matrix
    assert m.getElement(0, 0) == 1.0
    assert m.getElement(0, 2) == 2.5
    assert m.getElement(1, 1) == -3.0

    # Test on mutable view
    assert view.getElement(0, 0) == 1.0
    assert view.getElement(0, 2) == 2.5
    assert view.getElement(1, 1) == -3.0

    # Test on const view
    assert cview.getElement(0, 0) == 1.0
    assert cview.getElement(0, 2) == 2.5
    assert cview.getElement(1, 1) == -3.0


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_addElement(matrix_type: type[M]) -> None:
    """addElement accumulates values at the same position."""
    m = create_matrix(matrix_type, 1, 2, [(0, 0, 1.0), (0, 1, 2.0)])  # type: ignore[arg-type]

    # addElement on matrix
    m.addElement(0, 0, 10.0, 1.0)
    m.addElement(0, 0, 20.0, 1.0)
    assert m.getElement(0, 0) == 31.0  # 1.0 + 10.0 + 20.0

    # addElement on view modifies matrix
    view = m.getView()
    view.addElement(0, 0, 10.0, 1.0)
    assert m.getElement(0, 0) == 41.0  # 31.0 + 10.0
    assert view.getElement(0, 0) == 41.0


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_reset(matrix_type: type[M]) -> None:
    """reset() must zero out dimensions."""
    m = create_matrix(matrix_type, 2, 2, [(0, 0, 1.0)])  # type: ignore[arg-type]
    assert m.getRows() == 2
    m.reset()
    assert m.getRows() == 0
    assert m.getColumns() == 0


# ---------------------------------------------------------------------------
# RowView
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_rowview_basics(matrix_type: type[M]) -> None:
    """getRow returns a RowView with correct size/index/values."""
    m = create_matrix(matrix_type, 2, 3, [(0, 1, 4.0), (0, 2, 5.0), (1, 0, 6.0)])  # type: ignore[arg-type]

    row0 = m.getRow(0)
    assert isinstance(row0, SparseMatrixRowView)
    assert row0.getRowIndex() == 0
    # For CSR, getSize() == number of stored elements.
    # For Ellpack/SlicedEllpack, getSize() may be larger (uniform allocation).
    # Verify by finding our entries within the allocated row.
    found_01 = False
    found_02 = False
    for i in range(row0.getSize()):
        if row0.getColumnIndex(i) == 1 and row0.getValue(i) == 4.0:
            found_01 = True
        if row0.getColumnIndex(i) == 2 and row0.getValue(i) == 5.0:
            found_02 = True
    assert found_01
    assert found_02

    row1 = m.getRow(1)
    assert row1.getRowIndex() == 1
    found_10 = False
    for i in range(row1.getSize()):
        if row1.getColumnIndex(i) == 0 and row1.getValue(i) == 6.0:
            found_10 = True
    assert found_10


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_rowview_mutators(matrix_type: type[M]) -> None:
    """setValue / setColumnIndex / setElement on RowView modify underlying matrix."""
    m = create_matrix(matrix_type, 1, 2, [(0, 0, 1.0), (0, 1, 2.0)])  # type: ignore[arg-type]

    row = m.getRow(0)
    row.setValue(0, 10.0)
    assert m.getElement(0, 0) == 10.0

    # setColumnIndex changes the column index at a local position
    row.setColumnIndex(1, 0)
    assert row.getColumnIndex(1) == 0

    # setElement sets both column index and value at a local position
    row.setElement(0, 0, 7.0)
    assert row.getValue(0) == 7.0
    assert row.getColumnIndex(0) == 0


# ---------------------------------------------------------------------------
# vectorProduct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_vectorProduct(matrix_type: type[M]) -> None:
    """vectorProduct of identity matrix x vector equals the input vector."""
    n = 3
    m = identity_matrix(matrix_type, n)  # type: ignore[arg-type]

    in_vec = Vector[float](n)
    for i in range(n):
        in_vec[i] = float(i + 1)

    out_vec = Vector[float](n)
    m.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)

    for i in range(n):
        assert out_vec[i] == pytest.approx(float(i + 1))

    # Test through view
    out_vec_view = Vector[float](n)
    view = m.getView()
    view.vectorProduct(in_vec, out_vec_view, 1.0, 0.0, 0, 0)

    for i in range(n):
        assert out_vec_view[i] == pytest.approx(float(i + 1))


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_vectorProduct_with_factors(matrix_type: type[M]) -> None:
    """vectorProduct with matrixMult=2.0, outMult=3.0 on identity matrix."""
    n = 3
    m = identity_matrix(matrix_type, n)  # type: ignore[arg-type]

    in_vec = Vector[float](n)
    for i in range(n):
        in_vec[i] = float(i + 1)

    out_vec = Vector[float](n)
    for i in range(n):
        out_vec[i] = 10.0  # pre-fill

    # out = 2.0 * (I * in) + 3.0 * out
    m.vectorProduct(in_vec, out_vec, 2.0, 3.0, 0, 0)

    for i in range(n):
        expected_val = 2.0 * float(i + 1) + 3.0 * 10.0
        assert out_vec[i] == pytest.approx(expected_val)

    # Test through view
    out_vec_view = Vector[float](n)
    for i in range(n):
        out_vec_view[i] = 10.0

    view = m.getView()
    view.vectorProduct(in_vec, out_vec_view, 2.0, 3.0, 0, 0)

    for i in range(n):
        expected_val = 2.0 * float(i + 1) + 3.0 * 10.0
        assert out_vec_view[i] == pytest.approx(expected_val)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_equality(matrix_type: type[M]) -> None:
    """Two matrices with identical content are equal by ==."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    m1 = create_matrix(matrix_type, 2, 3, entries)  # type: ignore[arg-type]
    m2 = create_matrix(matrix_type, 2, 3, entries)  # type: ignore[arg-type]
    assert m1 == m2


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_inequality(matrix_type: type[M]) -> None:
    """Matrices with different content must not be equal."""
    m1 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (0, 2, 2.5)])  # type: ignore[arg-type]
    m2 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (0, 2, 9.9)])  # type: ignore[arg-type]
    assert m1 != m2

    # Different number of non-zero elements
    m3 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0)])  # type: ignore[arg-type]
    m4 = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (1, 1, 2.0)])  # type: ignore[arg-type]
    assert m3 != m4


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
    result = Vector_int(3)
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
    cl = Vector_int(3)
    m.getCompressedRowLengths(cl)
    assert cl.getSize() == 3
    assert cl[0] == 2
    assert cl[1] == 1
    assert cl[2] == 3

    # Same accessors through view
    view = m.getView()
    view_result = Vector_int(3)
    view.getRowCapacities(view_result)
    assert view_result.getSize() == 3
    assert view_result[0] >= 2
    assert view_result[1] >= 1
    assert view_result[2] >= 3

    assert view.getRowCapacity(0) >= 2

    view_cl = Vector_int(3)
    view.getCompressedRowLengths(view_cl)
    assert view_cl.getSize() == 3
    assert view_cl[0] == 2
    assert view_cl[1] == 1
    assert view_cl[2] == 3


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

    # View reports same counts
    view = m.getView()
    assert view.getNonzeroElementsCount() == 5
    assert view.getAllocatedElementsCount() >= 5


# ---------------------------------------------------------------------------
# Internal accessors (getValues, getColumnIndexes, getSegments)
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

    # Same accessors through view
    view = m.getView()
    view_values = view.getValues()
    assert view_values is not None
    assert view_values.getSize() > 0

    view_col_idx = view.getColumnIndexes()
    assert view_col_idx is not None
    assert view_col_idx.getSize() > 0

    view_segments = view.getSegments()
    assert view_segments is not None


# ---------------------------------------------------------------------------
# copySparseMatrix — all 6 conversion pairs
# ---------------------------------------------------------------------------


def test_copySparseMatrix_csr_to_ellpack_and_back() -> None:
    """CSR → Ellpack → CSR round-trip preserves content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 0, 4.0)]
    source = create_matrix(CSR, 3, 3, entries)

    dest_ell = Ellpack()
    dest_ell.setDimensions(3, 3)
    copySparseMatrix(dest_ell, source)
    for r, c, v in entries:
        assert dest_ell.getElement(r, c) == v

    dest_csr = CSR()
    dest_csr.setDimensions(3, 3)
    copySparseMatrix(dest_csr, dest_ell)
    for r, c, v in entries:
        assert dest_csr.getElement(r, c) == v
    assert dest_csr.getElement(1, 0) == 0.0
    assert dest_csr.getElement(2, 2) == 0.0
    assert dest_csr.getRows() == 3
    assert dest_csr.getColumns() == 3


def test_copySparseMatrix_csr_to_sliced_ellpack_and_back() -> None:
    """CSR → SlicedEllpack → CSR round-trip preserves content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 0, 4.0)]
    source = create_matrix(CSR, 3, 3, entries)

    dest_se = SlicedEllpack()
    dest_se.setDimensions(3, 3)
    copySparseMatrix(dest_se, source)
    for r, c, v in entries:
        assert dest_se.getElement(r, c) == v

    dest_csr = CSR()
    dest_csr.setDimensions(3, 3)
    copySparseMatrix(dest_csr, dest_se)
    for r, c, v in entries:
        assert dest_csr.getElement(r, c) == v
    assert dest_csr.getElement(1, 0) == 0.0
    assert dest_csr.getElement(2, 2) == 0.0
    assert dest_csr.getRows() == 3
    assert dest_csr.getColumns() == 3


def test_copySparseMatrix_ellpack_to_sliced_ellpack_and_back() -> None:
    """Ellpack → SlicedEllpack → Ellpack round-trip preserves content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 0, 4.0)]
    source = create_matrix(Ellpack, 3, 3, entries)  # type: ignore[arg-type]

    dest_se = SlicedEllpack()
    dest_se.setDimensions(3, 3)
    copySparseMatrix(dest_se, source)
    for r, c, v in entries:
        assert dest_se.getElement(r, c) == v

    dest_ell = Ellpack()
    dest_ell.setDimensions(3, 3)
    copySparseMatrix(dest_ell, dest_se)
    for r, c, v in entries:
        assert dest_ell.getElement(r, c) == v
    assert dest_ell.getElement(1, 0) == 0.0
    assert dest_ell.getElement(2, 2) == 0.0
    assert dest_ell.getRows() == 3
    assert dest_ell.getColumns() == 3


def test_copySparseMatrix_all_pairs() -> None:
    """All six conversion pairs preserve content (verified via getElement)."""
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
        copySparseMatrix(dst, src)
        for r, c, v in entries:
            assert dst.getElement(r, c) == v

        roundtrip = roundtrip_type()
        roundtrip.setDimensions(2, 2)
        copySparseMatrix(roundtrip, dst)
        for r, c, v in entries:
            assert roundtrip.getElement(r, c) == v
        assert roundtrip.getElement(0, 1) == 0.0
        assert roundtrip.getElement(1, 0) == 0.0
        assert roundtrip.getRows() == 2
        assert roundtrip.getColumns() == 2


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

    view = m.getView()
    s_view = str(view)
    assert isinstance(s_view, str)
    assert len(s_view) > 0

    cview = m.getConstView()
    s_cview = str(cview)
    assert isinstance(s_cview, str)
    assert len(s_cview) > 0


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
# setLike
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_setLike(matrix_type: type[M]) -> None:
    """setLike makes destination dimensions match source."""
    source = create_matrix(matrix_type, 3, 4, [(0, 0, 1.0)])  # type: ignore[arg-type]
    dest = matrix_type()
    dest.setLike(source)  # type: ignore[arg-type]
    assert dest.getRows() == source.getRows()
    assert dest.getColumns() == source.getColumns()


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


# ---------------------------------------------------------------------------
# Hypothesis: property-based vectorProduct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
@given(data=st.data())
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
    in_values = data.draw(st.lists(st.floats(allow_nan=False, allow_infinity=False, min_value=-10, max_value=10), min_size=n, max_size=n))

    in_vec = Vector[float](n)
    for i, val in enumerate(in_values):
        in_vec[i] = val

    out_vec = Vector[float](n)
    m.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)

    expected = dense @ np.array(in_values)
    for i in range(n):
        assert out_vec[i] == pytest.approx(expected[i], rel=1e-10)


# ---------------------------------------------------------------------------
# Subscript: SparseMatrixView resolution
# ---------------------------------------------------------------------------


def test_view_subscript_returns_correct_class() -> None:
    """Verify subscript syntax resolves to the correct view classes."""
    assert SparseMatrixView[float] is SV_CSR
    assert SparseMatrixView[float, Host, formats.CSR] is SV_CSR
    assert SparseMatrixView[float, Host, formats.Ellpack] is SV_Ellpack
    assert SparseMatrixView[float, Host, formats.SlicedEllpack] is SV_SlicedEllpack


def test_view_subscript_invalid_value_type() -> None:
    """Non-float value types must raise TypeError."""
    for bad_type in (int, complex, bool):
        with pytest.raises(TypeError):
            SparseMatrixView[bad_type]  # type: ignore[index, unused-ignore]


# ---------------------------------------------------------------------------
# getView / getConstView: correct types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_getView_returns_correct_type(matrix_type: type[M]) -> None:
    """getView returns the correct mutable view type for each format."""
    m = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0)])  # type: ignore[arg-type]
    view = m.getView()
    if matrix_type is CSR:
        assert isinstance(view, SV_CSR)
    elif matrix_type is Ellpack:
        assert isinstance(view, SV_Ellpack)
    else:
        assert isinstance(view, SV_SlicedEllpack)


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_getConstView_returns_correct_type(matrix_type: type[M]) -> None:
    """getConstView returns the correct const view type for each format."""
    m = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0)])  # type: ignore[arg-type]
    view = m.getConstView()
    if matrix_type is CSR:
        assert isinstance(view, SV_CSR_const)
    elif matrix_type is Ellpack:
        assert isinstance(view, SV_Ellpack_const)
    else:
        assert isinstance(view, SV_SlicedEllpack_const)


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_view_dimensions_match_parent(matrix_type: type[M]) -> None:
    """View dimensions match the parent matrix dimensions."""
    m = create_matrix(matrix_type, 3, 4, [(0, 0, 1.0), (2, 3, 2.0)])  # type: ignore[arg-type]
    view = m.getView()
    assert view.getRows() == 3
    assert view.getColumns() == 4

    cview = m.getConstView()
    assert cview.getRows() == 3
    assert cview.getColumns() == 4


# ---------------------------------------------------------------------------
# Reference semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_reference_semantics(matrix_type: type[M]) -> None:
    """View and parent share the same underlying data."""
    m = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0), (1, 1, 2.0)])  # type: ignore[arg-type]
    view = m.getView()

    # setElement through view modifies parent
    view.setElement(0, 0, 99.0)
    assert m.getElement(0, 0) == 99.0

    # setElement through parent is visible through view
    m.setElement(0, 0, 88.0)
    assert view.getElement(0, 0) == 88.0


# ---------------------------------------------------------------------------
# Const view: mutation methods raise AttributeError
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_const_view_setElement_raises(matrix_type: type[M]) -> None:
    """const view has no setElement method."""
    m = create_matrix(matrix_type, 2, 2, [(0, 0, 1.0)])  # type: ignore[arg-type]
    cview = m.getConstView()
    with pytest.raises(AttributeError):
        cview.setElement(0, 0, 99.0)  # type: ignore[attr-defined]


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_const_view_addElement_raises(matrix_type: type[M]) -> None:
    """const view has no addElement method."""
    m = create_matrix(matrix_type, 2, 2, [(0, 0, 1.0)])  # type: ignore[arg-type]
    cview = m.getConstView()
    with pytest.raises(AttributeError):
        cview.addElement(0, 0, 1.0)  # type: ignore[attr-defined]


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_const_view_getRow_raises(matrix_type: type[M]) -> None:
    """const view has no getRow method."""
    m = create_matrix(matrix_type, 2, 2, [(0, 0, 1.0)])  # type: ignore[arg-type]
    cview = m.getConstView()
    with pytest.raises(AttributeError):
        cview.getRow(0)  # type: ignore[attr-defined]


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_const_view_vectorProduct_raises(matrix_type: type[M]) -> None:
    """const view has no vectorProduct method."""
    m = identity_matrix(matrix_type, 3)  # type: ignore[arg-type]
    cview = m.getConstView()
    in_vec = Vector[float](3)
    out_vec = Vector[float](3)
    with pytest.raises(AttributeError):
        cview.vectorProduct(in_vec, out_vec)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_bounds(matrix_type: type[M]) -> None:
    """Out-of-bounds access raises IndexError on matrix, view, and const view."""
    m = create_matrix(matrix_type, 2, 3, [(0, 0, 1.0)])  # type: ignore[arg-type]
    view = m.getView()
    cview = m.getConstView()

    for target in (m, view, cview):
        with pytest.raises(IndexError):
            target.getElement(5, 0)
        with pytest.raises(IndexError):
            target.getElement(0, 5)

    for target in (m, view):
        with pytest.raises(IndexError):
            target.setElement(5, 0, 1.0)
        with pytest.raises(IndexError):
            target.setElement(0, 5, 1.0)
        with pytest.raises(IndexError):
            target.getRow(5)

    # addElement bounds on matrix and view
    with pytest.raises(IndexError):
        m.addElement(5, 0, 1.0)
    with pytest.raises(IndexError):
        m.addElement(0, 5, 1.0)
    with pytest.raises(IndexError):
        view.addElement(5, 0, 1.0)
    with pytest.raises(IndexError):
        view.addElement(0, 5, 1.0)


# ---------------------------------------------------------------------------
# getRow through view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_view_getRow_returns_rowview(matrix_type: type[M]) -> None:
    """getRow through view returns a RowView bound to the view's data."""
    m = create_matrix(matrix_type, 2, 3, [(0, 1, 4.0), (0, 2, 5.0), (1, 0, 6.0)])  # type: ignore[arg-type]
    view = m.getView()
    row = view.getRow(0)

    assert isinstance(row, SparseMatrixRowView)
    assert row.getRowIndex() == 0

    # Verify entries are reachable
    found_01 = False
    found_02 = False
    for i in range(row.getSize()):
        if row.getColumnIndex(i) == 1 and row.getValue(i) == 4.0:
            found_01 = True
        if row.getColumnIndex(i) == 2 and row.getValue(i) == 5.0:
            found_02 = True
    assert found_01
    assert found_02


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_view_getRow_mutations_propagate(matrix_type: type[M]) -> None:
    """Mutating via view.getRow modifies the parent matrix."""
    m = create_matrix(matrix_type, 1, 2, [(0, 0, 1.0), (0, 1, 2.0)])  # type: ignore[arg-type]
    view = m.getView()
    row = view.getRow(0)
    row.setValue(0, 10.0)
    assert m.getElement(0, 0) == 10.0


# ---------------------------------------------------------------------------
# bind: rebind view to another view
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_view_bind_rebinds_to_other_view(matrix_type: type[M]) -> None:
    """bind(view) makes the view refer to another view's data."""
    entries1 = [(0, 0, 1.0), (0, 1, 2.0)]
    entries2 = [(0, 0, 99.0), (0, 1, 100.0)]
    m1 = create_matrix(matrix_type, 2, 3, entries1)  # type: ignore[arg-type]
    m2 = create_matrix(matrix_type, 2, 3, entries2)  # type: ignore[arg-type]

    view1 = m1.getView()
    view2 = m2.getView()

    # Initially, view1 sees m1's data
    assert view1.getElement(0, 0) == 1.0

    # Rebind view1 to view2's data
    view1.bind(view2)
    assert view1.getElement(0, 0) == 99.0
    assert view1.getElement(0, 1) == 100.0


# ---------------------------------------------------------------------------
# Base class inheritance and memory management
# ---------------------------------------------------------------------------


def test_isinstance_sparse_matrix_base() -> None:
    """SparseMatrix and SparseMatrixView inherit from SparseMatrixBase."""
    from pytnl._matrices import SparseMatrixBase_float_CSR  # type: ignore[attr-defined, unused-ignore]  # noqa: PLC0415

    m = SparseMatrix[float, Host]()
    m.setDimensions(2, 2)
    caps = Vector_int(2)
    caps[0] = 1
    caps[1] = 1
    m.setRowCapacities(caps)
    m.setElement(0, 0, 1.0)
    m.setElement(1, 1, 2.0)
    assert isinstance(m, SparseMatrixBase_float_CSR)
    view = m.getView()
    assert isinstance(view, SparseMatrixBase_float_CSR)


def test_inherited_get_values_reference_internal() -> None:
    """getValues() on a view keeps the view alive via reference_internal."""
    import gc  # noqa: PLC0415

    m = SparseMatrix[float, Host]()
    m.setDimensions(2, 2)
    caps = Vector_int(2)
    caps[0] = 1
    caps[1] = 1
    m.setRowCapacities(caps)
    m.setElement(0, 0, 1.0)
    m.setElement(1, 1, 2.0)
    view = m.getView()
    vals = view.getValues()
    del view
    gc.collect()
    # vals should still be accessible because reference_internal keeps the view alive
    assert vals.getSize() == 2


def test_bind_keep_alive_temporary_view() -> None:
    """bind() keeps a temporary source view alive via keep_alive."""
    import gc  # noqa: PLC0415

    m = SparseMatrix[float, Host]()
    m.setDimensions(2, 2)
    caps = Vector_int(2)
    caps[0] = 1
    caps[1] = 1
    m.setRowCapacities(caps)
    m.setElement(0, 0, 42.0)

    target = SparseMatrix[float, Host]()
    target.setDimensions(2, 2)
    target_view = target.getView()

    # bind to a temporary view — without keep_alive, this temporary would be GC'd
    target_view.bind(m.getView())
    gc.collect()

    assert target_view.getElement(0, 0) == 42.0


# ---------------------------------------------------------------------------
# isBinary / isSymmetric (def_static on MatrixBase, inherited by owners & views)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("matrix_type", matrix_types)
def test_is_binary_is_symmetric(matrix_type: type[M]) -> None:
    """isBinary/isSymmetric return False for GeneralMatrix on owner and view."""
    m = create_matrix(matrix_type, 2, 2, [(0, 0, 1.0), (1, 1, 2.0)])  # type: ignore[arg-type]

    # Instance-level calls on the owning matrix
    assert m.isBinary() is False
    assert m.isSymmetric() is False

    # Static calls on the class itself
    assert matrix_type.isBinary() is False
    assert matrix_type.isSymmetric() is False

    # Instance-level calls on a mutable view
    view = m.getView()
    assert view.isBinary() is False
    assert view.isSymmetric() is False

    # Instance-level calls on a const view
    cview = m.getConstView()
    assert cview.isBinary() is False
    assert cview.isSymmetric() is False

import os
import tempfile

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import pytnl._matrices
from pytnl._containers import Vector_int
from pytnl.containers import Vector
from pytnl.devices import Host
from pytnl.matrices import DenseMatrix, DenseMatrixRowView, DenseMatrixView, ElementsOrganization

# ---------------------------------------------------------------------------
# Single host matrix type (no format parameter — only value_type + device_type)
# ---------------------------------------------------------------------------

DM = pytnl._matrices.DenseMatrix_float_RowMajor
DV = pytnl._matrices.DenseMatrixView_float_RowMajor
DV_const = pytnl._matrices.DenseMatrixView_float_RowMajor_const


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_matrix(
    rows: int,
    cols: int,
    entries: list[tuple[int, int, float]],
) -> pytnl._matrices.DenseMatrix_float_RowMajor:
    """Create a dense matrix filled with the given entries.

    ``entries`` is a list of ``(row, col, value)`` tuples.
    All elements default to 0.0; only the listed entries are set.
    No ``setRowCapacities`` call is needed for dense matrices.
    """
    m = DenseMatrix[float]()
    m.setDimensions(rows, cols)
    m.setValue(0.0)
    for r, c, v in entries:
        m.setElement(r, c, v)
    return m


def identity_matrix(size: int) -> pytnl._matrices.DenseMatrix_float_RowMajor:
    """Create a size x size identity matrix."""
    entries = [(i, i, 1.0) for i in range(size)]
    return create_matrix(size, size, entries)


# ---------------------------------------------------------------------------
# Subscript / pythonization: correct class resolution
# ---------------------------------------------------------------------------


def test_subscript_returns_correct_class() -> None:
    """Verify subscript syntax resolves to the correct C++ class."""
    assert DenseMatrix[float] is DM
    assert DenseMatrix[float, Host] is DM


def test_subscript_default_device() -> None:
    """DenseMatrix[float] defaults to Host device."""
    assert DenseMatrix[float] is DenseMatrix[float, Host]


def test_subscript_invalid_value_type() -> None:
    """Non-float value types must raise TypeError."""
    for bad_type in (int, complex, bool):
        with pytest.raises(TypeError):
            DenseMatrix[bad_type]  # type: ignore[index, unused-ignore]


# ---------------------------------------------------------------------------
# Construction and basic properties
# ---------------------------------------------------------------------------


def test_construction_empty() -> None:
    """Default-constructed matrix has zero dimensions."""
    m = DM()
    assert m.getRows() == 0
    assert m.getColumns() == 0


def test_construction_with_dimensions() -> None:
    """Constructor with (rows, cols) sets dimensions directly."""
    m = DM(3, 4)
    assert m.getRows() == 3
    assert m.getColumns() == 4


def test_setDimensions() -> None:
    """setDimensions must update getRows / getColumns."""
    m = DM()
    m.setDimensions(3, 4)
    assert m.getRows() == 3
    assert m.getColumns() == 4


# ---------------------------------------------------------------------------
# Element access
# ---------------------------------------------------------------------------


def test_setElement_getElement() -> None:
    """setElement / getElement round-trip on matrix, view, and const view."""
    m = create_matrix(2, 3, [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)])

    view = m.getView()
    cview = m.getConstView()

    assert m.getElement(0, 0) == 1.0
    assert m.getElement(0, 2) == 2.5
    assert m.getElement(1, 1) == -3.0
    # Unset elements default to 0.0 (dense stores all elements)
    assert m.getElement(0, 1) == 0.0
    assert m.getElement(1, 0) == 0.0
    assert m.getElement(1, 2) == 0.0

    assert view.getElement(0, 0) == 1.0
    assert view.getElement(0, 2) == 2.5
    assert view.getElement(1, 1) == -3.0
    assert view.getElement(0, 1) == 0.0
    assert view.getElement(1, 0) == 0.0

    assert cview.getElement(0, 0) == 1.0
    assert cview.getElement(0, 2) == 2.5
    assert cview.getElement(1, 1) == -3.0
    assert cview.getElement(0, 1) == 0.0
    assert cview.getElement(1, 0) == 0.0

    # setElement on matrix is visible through view and const view
    m.setElement(0, 1, 7.0)
    assert m.getElement(0, 1) == 7.0
    assert view.getElement(0, 1) == 7.0
    assert cview.getElement(0, 1) == 7.0

    # setElement on view modifies the parent matrix
    view.setElement(1, 0, 8.0)
    assert view.getElement(1, 0) == 8.0
    assert m.getElement(1, 0) == 8.0
    assert cview.getElement(1, 0) == 8.0


def test_addElement() -> None:
    """addElement accumulates values on matrix and view."""
    m = create_matrix(1, 2, [(0, 0, 1.0), (0, 1, 2.0)])

    # addElement on the owning matrix
    m.addElement(0, 0, 10.0, 1.0)
    m.addElement(0, 0, 20.0, 1.0)
    assert m.getElement(0, 0) == 31.0  # 1.0 + 10.0 + 20.0

    # addElement on view modifies the parent matrix
    view = m.getView()
    view.addElement(0, 0, 10.0, 1.0)
    assert m.getElement(0, 0) == 41.0  # 31.0 + 10.0
    assert view.getElement(0, 0) == 41.0

    view.addElement(0, 0, 20.0, 1.0)
    assert m.getElement(0, 0) == 61.0
    assert view.getElement(0, 0) == 61.0


def test_setValue_all_elements() -> None:
    """setValue sets ALL elements to the given value (DenseMatrix-specific)."""
    m = DM(2, 3)
    m.setValue(5.0)
    for r in range(2):
        for c in range(3):
            assert m.getElement(r, c) == 5.0

    # setValue through view sets all elements in the parent matrix
    view = m.getView()
    view.setValue(7.0)
    for r in range(2):
        for c in range(3):
            assert m.getElement(r, c) == 7.0
            assert view.getElement(r, c) == 7.0


def test_reset() -> None:
    """reset() must zero out dimensions."""
    m = create_matrix(2, 2, [(0, 0, 1.0)])
    assert m.getRows() == 2
    m.reset()
    assert m.getRows() == 0
    assert m.getColumns() == 0


# ---------------------------------------------------------------------------
# RowView
# ---------------------------------------------------------------------------


def test_rowview_basics() -> None:
    """getRow returns a RowView with correct size/index/values."""
    m = create_matrix(2, 3, [(0, 1, 4.0), (0, 2, 5.0), (1, 0, 6.0)])

    row0 = m.getRow(0)
    assert isinstance(row0, DenseMatrixRowView)
    assert row0.getRowIndex() == 0
    # For dense, getSize() == number of columns
    assert row0.getSize() == 3
    # getValue takes the ACTUAL column index (not local index)
    assert row0.getValue(0) == 0.0
    assert row0.getValue(1) == 4.0
    assert row0.getValue(2) == 5.0
    # getColumnIndex returns localIdx (identity for dense)
    assert row0.getColumnIndex(0) == 0
    assert row0.getColumnIndex(1) == 1
    assert row0.getColumnIndex(2) == 2

    row1 = m.getRow(1)
    assert row1.getRowIndex() == 1
    assert row1.getSize() == 3
    assert row1.getValue(0) == 6.0
    assert row1.getValue(1) == 0.0
    assert row1.getValue(2) == 0.0


def test_rowview_mutators() -> None:
    """setValue / setElement on RowView modify underlying matrix."""
    m = create_matrix(1, 2, [(0, 0, 1.0), (0, 1, 2.0)])

    row = m.getRow(0)
    # setValue takes the ACTUAL column index
    row.setValue(0, 10.0)
    assert m.getElement(0, 0) == 10.0

    # setElement(localIdx, column, value) — localIdx is unused for dense
    row.setElement(0, 1, 7.0)
    assert row.getValue(1) == 7.0
    assert m.getElement(0, 1) == 7.0


# ---------------------------------------------------------------------------
# vectorProduct
# ---------------------------------------------------------------------------


def test_vectorProduct() -> None:
    """vectorProduct of identity matrix x vector equals the input vector."""
    n = 3
    m = identity_matrix(n)

    in_vec = Vector[float](n)
    for i in range(n):
        in_vec[i] = float(i + 1)

    out_vec = Vector[float](n)
    m.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)

    for i in range(n):
        assert out_vec[i] == pytest.approx(float(i + 1))

    # vectorProduct through view
    out_vec2 = Vector[float](n)
    view = m.getView()
    view.vectorProduct(in_vec, out_vec2, 1.0, 0.0, 0, 0)
    for i in range(n):
        assert out_vec2[i] == pytest.approx(float(i + 1))


def test_vectorProduct_with_factors() -> None:
    """vectorProduct with matrixMult=2.0, outMult=3.0 on identity matrix."""
    n = 3
    m = identity_matrix(n)

    in_vec = Vector[float](n)
    for i in range(n):
        in_vec[i] = float(i + 1)

    out_vec = Vector[float](n)
    for i in range(n):
        out_vec[i] = 10.0  # pre-fill

    # out = 2.0 * (I * in) + 3.0 * out
    m.vectorProduct(in_vec, out_vec, 2.0, 3.0, 0, 0)

    for i in range(n):
        expected = 2.0 * float(i + 1) + 3.0 * 10.0
        assert out_vec[i] == pytest.approx(expected)

    # vectorProduct with factors through view
    out_vec2 = Vector[float](n)
    for i in range(n):
        out_vec2[i] = 10.0
    view = m.getView()
    view.vectorProduct(in_vec, out_vec2, 2.0, 3.0, 0, 0)
    for i in range(n):
        expected = 2.0 * float(i + 1) + 3.0 * 10.0
        assert out_vec2[i] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def test_equality() -> None:
    """Two matrices with identical content are equal by ==."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    m1 = create_matrix(2, 3, entries)
    m2 = create_matrix(2, 3, entries)
    assert m1 == m2


def test_inequality() -> None:
    """Matrices with different content must not be equal."""
    m1 = create_matrix(2, 3, [(0, 0, 1.0), (0, 2, 2.5)])
    m2 = create_matrix(2, 3, [(0, 0, 1.0), (0, 2, 9.9)])
    assert m1 != m2

    # Different number of non-zero elements
    m3 = create_matrix(2, 3, [(0, 0, 1.0)])
    m4 = create_matrix(2, 3, [(0, 0, 1.0), (1, 1, 2.0)])
    assert m3 != m4


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_getSerializationType() -> None:
    """getSerializationType returns a non-empty string."""
    serial_type = DM.getSerializationType()
    assert isinstance(serial_type, str)
    assert len(serial_type) > 0


def test_save_load_roundtrip() -> None:
    """save / load round-trip must preserve content."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    m1 = create_matrix(2, 3, entries)

    with tempfile.NamedTemporaryFile(mode="w+b", delete=False) as tmpfile:
        filename = tmpfile.name

    try:
        m1.save(str(filename))
        m2 = DM()
        m2.load(str(filename))
        assert m2.getRows() == m1.getRows()
        assert m2.getColumns() == m1.getColumns()
        assert m2 == m1
    finally:
        os.unlink(filename)


# ---------------------------------------------------------------------------
# Row capacities and compressed lengths
# ---------------------------------------------------------------------------


def test_row_capacities() -> None:
    """getRowCapacities / getRowCapacity / getCompressedRowLengths for dense (matrix and view)."""
    entries = [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (2, 0, 4.0), (2, 1, 5.0), (2, 2, 6.0)]
    m = create_matrix(3, 4, entries)

    # Matrix
    result = Vector_int(3)
    m.getRowCapacities(result)
    assert result.getSize() == 3
    # For dense, row capacity = number of columns
    assert result[0] == 4
    assert result[1] == 4
    assert result[2] == 4

    assert m.getRowCapacity(0) == 4
    assert m.getRowCapacity(1) == 4
    assert m.getRowCapacity(2) == 4

    cl = Vector_int(3)
    m.getCompressedRowLengths(cl)
    assert cl.getSize() == 3
    assert cl[0] == 2
    assert cl[1] == 1
    assert cl[2] == 3

    # View
    view = m.getView()
    result2 = Vector_int(3)
    view.getRowCapacities(result2)
    assert result2.getSize() == 3
    assert result2[0] == 4
    assert result2[1] == 4
    assert result2[2] == 4

    assert view.getRowCapacity(0) == 4
    assert view.getRowCapacity(1) == 4
    assert view.getRowCapacity(2) == 4

    cl2 = Vector_int(3)
    view.getCompressedRowLengths(cl2)
    assert cl2.getSize() == 3
    assert cl2[0] == 2
    assert cl2[1] == 1
    assert cl2[2] == 3


# ---------------------------------------------------------------------------
# Nonzero and allocation counts
# ---------------------------------------------------------------------------


def test_nonzero_counts() -> None:
    """getNonzeroElementsCount and getAllocatedElementsCount for matrix and view."""
    entries = [(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 1, 5.0)]
    m = create_matrix(3, 3, entries)

    assert m.getNonzeroElementsCount() == 5
    # For dense, allocated = rows * cols (all elements are allocated)
    assert m.getAllocatedElementsCount() == 9

    view = m.getView()
    assert view.getNonzeroElementsCount() == 5
    assert view.getAllocatedElementsCount() == 9

    # Standalone: allocated = rows * cols for any empty dense matrix
    m2 = DM(3, 4)
    assert m2.getAllocatedElementsCount() == 12


# ---------------------------------------------------------------------------
# Internal accessors (getValues)
# ---------------------------------------------------------------------------


def test_getValues() -> None:
    """getValues returns the internal values vector with rows * cols elements."""
    m = create_matrix(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])

    values = m.getValues()
    assert values is not None
    # Dense matrix stores rows * cols elements
    assert values.getSize() == 4

    # getValues through view
    view = m.getView()
    view_values = view.getValues()
    assert view_values is not None
    assert view_values.getSize() == 4


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


def test_str_output() -> None:
    """str(matrix), str(view), str(const_view) return non-empty strings."""
    m = create_matrix(2, 3, [(0, 0, 1.0), (1, 2, 3.0)])
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


def test_assign() -> None:
    """assign copies content; result equals source."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    source = create_matrix(2, 3, entries)

    dest = DM()
    dest.assign(source)
    assert dest == source


# ---------------------------------------------------------------------------
# setLike
# ---------------------------------------------------------------------------


def test_setLike() -> None:
    """setLike makes destination dimensions match source."""
    source = create_matrix(3, 4, [(0, 0, 1.0)])
    dest = DM()
    dest.setLike(source)
    assert dest.getRows() == source.getRows()
    assert dest.getColumns() == source.getColumns()


# ---------------------------------------------------------------------------
# Hypothesis: property-based element round-trip
# ---------------------------------------------------------------------------


@st.composite
def dense_entries_strategy(
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


@given(data=st.data())
def test_property_element_roundtrip(data: st.DataObject) -> None:
    """Property: set/get element round-trip correctly for random dense patterns."""
    rows = data.draw(st.integers(min_value=2, max_value=6))
    cols = data.draw(st.integers(min_value=2, max_value=6))
    entries = data.draw(dense_entries_strategy(rows, cols, max_elements=min(rows * cols, 10)))

    m = create_matrix(rows, cols, entries)

    for r, c, expected in entries:
        assert m.getElement(r, c) == expected


# ---------------------------------------------------------------------------
# Hypothesis: property-based vectorProduct
# ---------------------------------------------------------------------------


@given(data=st.data())
def test_property_vectorProduct(data: st.DataObject) -> None:
    """Property: dense matrix-vector product matches numpy computation."""
    n = data.draw(st.integers(min_value=2, max_value=5))
    entries = data.draw(dense_entries_strategy(n, n, max_elements=n * n))

    m = create_matrix(n, n, entries)

    # Build dense numpy matrix for reference
    dense = np.zeros((n, n), dtype=np.float64)
    for r, c, v in entries:
        dense[r, c] = v  # dense: use = (entries are unique, no accumulation)

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
# DLPack interoperability
# ---------------------------------------------------------------------------


def test_dlpack_device() -> None:
    """__dlpack_device__ returns CPU device for Host matrices and views."""
    m = DM(2, 3)
    device, _ = m.__dlpack_device__()
    assert device == 1

    view = m.getView()
    view_device, _ = view.__dlpack_device__()
    assert view_device == 1


def test_dlpack_to_numpy_shape_dtype() -> None:
    """DLPack export produces a numpy array with correct shape and dtype."""
    m = create_matrix(3, 4, [(0, 0, 1.0), (1, 2, 3.5), (2, 3, -2.0)])
    arr = np.from_dlpack(m)
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float64

    view = m.getView()
    view_arr = np.from_dlpack(view)
    assert view_arr.shape == (3, 4)
    assert view_arr.dtype == np.float64


def test_dlpack_to_numpy_values() -> None:
    """DLPack export preserves matrix element values."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 3, 4.0)]
    m = create_matrix(3, 4, entries)
    arr = np.from_dlpack(m)

    for r, c, v in entries:
        assert arr[r, c] == v
    assert arr[0, 1] == 0.0
    assert arr[1, 0] == 0.0

    view = m.getView()
    view_arr = np.from_dlpack(view)
    for r, c, v in entries:
        assert view_arr[r, c] == v
    assert view_arr[0, 1] == 0.0
    assert view_arr[1, 0] == 0.0


def test_dlpack_zero_copy() -> None:
    """DLPack export is zero-copy — modifying numpy array modifies the matrix."""
    m = create_matrix(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])

    # Matrix DLPack
    arr = np.from_dlpack(m)
    arr[0, 1] = 99.0
    assert m.getElement(0, 1) == 99.0

    # View DLPack is also zero-copy — modifies the parent
    view = m.getView()
    view_arr = np.from_dlpack(view)
    view_arr[1, 0] = 42.0
    assert m.getElement(1, 0) == 42.0
    assert view.getElement(1, 0) == 42.0


def test_dlpack_numpy_roundtrip() -> None:
    """numpy → DLPack import into DenseMatrix via getValues view."""
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    m = DM(arr.shape[0], arr.shape[1])
    values = m.getValues()
    np_view = np.from_dlpack(values)
    np_view[:] = arr.flatten()
    for r in range(2):
        for c in range(3):
            assert m.getElement(r, c) == arr[r, c]


def test_dlpack_strides_rowmajor() -> None:
    """Host DenseMatrix and DenseMatrixView (RowMajor) have C-contiguous strides."""
    rows, cols = 3, 4
    m = DM(rows, cols)

    arr = np.from_dlpack(m)
    assert arr.flags["C_CONTIGUOUS"]
    assert arr.strides == (cols * arr.itemsize, arr.itemsize)

    view = m.getView()
    view_arr = np.from_dlpack(view)
    assert view_arr.flags["C_CONTIGUOUS"]
    assert view_arr.strides == (cols * view_arr.itemsize, view_arr.itemsize)


# ---------------------------------------------------------------------------
# Subscript: DenseMatrixView resolution
# ---------------------------------------------------------------------------


def test_view_subscript_returns_correct_class() -> None:
    """Verify subscript syntax resolves to the correct view classes."""
    assert DenseMatrixView[float] is DV


def test_view_subscript_invalid_value_type() -> None:
    """Non-float value types must raise TypeError."""
    for bad_type in (int, complex, bool):
        with pytest.raises(TypeError):
            DenseMatrixView[bad_type]  # type: ignore[index, unused-ignore]


# ---------------------------------------------------------------------------
# getView / getConstView: correct types and reference semantics
# ---------------------------------------------------------------------------


def test_getView_returns_correct_type() -> None:
    """getView returns the correct mutable view type."""
    m = create_matrix(2, 3, [(0, 0, 1.0)])
    view = m.getView()
    assert isinstance(view, DV)


def test_getConstView_returns_correct_type() -> None:
    """getConstView returns the correct const view type."""
    m = create_matrix(2, 3, [(0, 0, 1.0)])
    view = m.getConstView()
    assert isinstance(view, DV_const)


def test_view_dimensions_match_parent() -> None:
    """View dimensions match the parent matrix dimensions."""
    m = create_matrix(3, 4, [(0, 0, 1.0), (2, 3, 2.0)])
    view = m.getView()
    assert view.getRows() == 3
    assert view.getColumns() == 4

    cview = m.getConstView()
    assert cview.getRows() == 3
    assert cview.getColumns() == 4


# ---------------------------------------------------------------------------
# Reference semantics: view and parent share data
# ---------------------------------------------------------------------------


def test_reference_semantics() -> None:
    """View and parent share data — modifications in either are visible in the other."""
    m = create_matrix(2, 3, [(0, 0, 1.0), (1, 1, 2.0)])
    view = m.getView()

    # setElement through view modifies the parent matrix
    view.setElement(0, 0, 99.0)
    assert m.getElement(0, 0) == 99.0

    # Modifying parent via setElement is visible through the view
    m.setElement(0, 0, 88.0)
    assert view.getElement(0, 0) == 88.0


# ---------------------------------------------------------------------------
# Const view: mutation methods raise AttributeError
# ---------------------------------------------------------------------------


def test_const_view_setElement_raises() -> None:
    """const view has no setElement method."""
    m = create_matrix(2, 2, [(0, 0, 1.0)])
    cview = m.getConstView()
    with pytest.raises(AttributeError):
        cview.setElement(0, 0, 99.0)  # type: ignore[attr-defined]


def test_const_view_addElement_raises() -> None:
    """const view has no addElement method."""
    m = create_matrix(2, 2, [(0, 0, 1.0)])
    cview = m.getConstView()
    with pytest.raises(AttributeError):
        cview.addElement(0, 0, 1.0)  # type: ignore[attr-defined]


def test_const_view_getRow_raises() -> None:
    """const view has no getRow method."""
    m = create_matrix(2, 2, [(0, 0, 1.0)])
    cview = m.getConstView()
    with pytest.raises(AttributeError):
        cview.getRow(0)  # type: ignore[attr-defined]


def test_const_view_setValue_raises() -> None:
    """const view has no setValue method."""
    m = create_matrix(2, 2, [(0, 0, 1.0)])
    cview = m.getConstView()
    with pytest.raises(AttributeError):
        cview.setValue(99.0)  # type: ignore[attr-defined]


def test_const_view_vectorProduct_raises() -> None:
    """const view has no vectorProduct method."""
    m = identity_matrix(3)
    cview = m.getConstView()
    in_vec = Vector[float](3)
    out_vec = Vector[float](3)
    with pytest.raises(AttributeError):
        cview.vectorProduct(in_vec, out_vec)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Bounds checking
# ---------------------------------------------------------------------------


def test_bounds_checking() -> None:
    """Bounds checking on matrix and view raises IndexError."""
    m = create_matrix(2, 3, [(0, 0, 1.0)])

    # Matrix bounds — getElement and setElement
    with pytest.raises(IndexError):
        m.getElement(5, 0)
    with pytest.raises(IndexError):
        m.getElement(0, 5)
    with pytest.raises(IndexError):
        m.setElement(5, 0, 1.0)
    with pytest.raises(IndexError):
        m.setElement(0, 5, 1.0)

    # View bounds — getElement, setElement, addElement, getRow
    view = m.getView()
    with pytest.raises(IndexError):
        view.getElement(5, 0)
    with pytest.raises(IndexError):
        view.getElement(0, 5)
    with pytest.raises(IndexError):
        view.setElement(5, 0, 1.0)
    with pytest.raises(IndexError):
        view.setElement(0, 5, 1.0)
    with pytest.raises(IndexError):
        view.addElement(5, 0, 1.0)
    with pytest.raises(IndexError):
        view.addElement(0, 5, 1.0)
    with pytest.raises(IndexError):
        view.getRow(5)


# ---------------------------------------------------------------------------
# getRow through view
# ---------------------------------------------------------------------------


def test_view_getRow_returns_rowview() -> None:
    """getRow through view returns a RowView bound to the view's data."""
    m = create_matrix(2, 3, [(0, 1, 4.0), (0, 2, 5.0), (1, 0, 6.0)])
    view = m.getView()
    row = view.getRow(0)

    assert isinstance(row, DenseMatrixRowView)
    assert row.getRowIndex() == 0
    assert row.getSize() == 3
    assert row.getValue(0) == 0.0
    assert row.getValue(1) == 4.0
    assert row.getValue(2) == 5.0


def test_view_getRow_mutations_propagate() -> None:
    """Mutating via view.getRow modifies the parent matrix."""
    m = create_matrix(1, 2, [(0, 0, 1.0), (0, 1, 2.0)])
    view = m.getView()
    row = view.getRow(0)
    row.setValue(0, 10.0)
    assert m.getElement(0, 0) == 10.0


# ---------------------------------------------------------------------------
# bind: rebind view to another view
# ---------------------------------------------------------------------------


def test_view_bind_rebinds_to_other_view() -> None:
    """bind(view) makes the view refer to another view's data."""
    m1 = create_matrix(2, 3, [(0, 0, 1.0), (0, 1, 2.0)])
    m2 = create_matrix(2, 3, [(0, 0, 99.0), (0, 1, 100.0)])

    view1 = m1.getView()
    view2 = m2.getView()

    # Initially, view1 sees m1's data
    assert view1.getElement(0, 0) == 1.0

    # Rebind view1 to view2's data
    view1.bind(view2)
    assert view1.getElement(0, 0) == 99.0
    assert view1.getElement(0, 1) == 100.0


# ---------------------------------------------------------------------------
# DLPack: const view
# ---------------------------------------------------------------------------


def test_const_view_dlpack_read_only() -> None:
    """Const view exports a read-only DLPack array with correct values."""
    m = DM(2, 3)
    m.setElement(0, 0, 1.0)
    m.setElement(0, 1, 2.0)
    m.setElement(1, 0, 3.0)
    m.setElement(1, 2, 4.0)
    cview = m.getConstView()

    arr = np.from_dlpack(cview)
    assert arr.shape == (2, 3)
    assert arr.dtype == np.float64
    assert arr[0, 0] == 1.0
    assert arr[0, 1] == 2.0
    assert arr[1, 0] == 3.0
    assert arr[1, 2] == 4.0
    # numpy ≥ 2.0 honors the DLPack read-only flag
    assert not arr.flags.writeable  # spellchecker:disable-line


# ---------------------------------------------------------------------------
# Base class inheritance and memory management
# ---------------------------------------------------------------------------


def test_isinstance_dense_matrix_base() -> None:
    """DenseMatrix and DenseMatrixView inherit from DenseMatrixBase."""
    from pytnl._matrices import DenseMatrixBase_float_RowMajor  # type: ignore[attr-defined, unused-ignore]  # noqa: PLC0415

    m = DenseMatrix[float]()
    m.setDimensions(2, 2)
    m.setValue(5.0)
    assert isinstance(m, DenseMatrixBase_float_RowMajor)
    view = m.getView()
    assert isinstance(view, DenseMatrixBase_float_RowMajor)


def test_inherited_get_values_reference_internal() -> None:
    """getValues() on a view keeps the view alive via reference_internal."""
    import gc  # noqa: PLC0415

    m = DenseMatrix[float]()
    m.setDimensions(2, 2)
    m.setValue(5.0)
    view = m.getView()
    vals = view.getValues()
    del view
    gc.collect()
    # vals should still be accessible because reference_internal keeps the view alive
    assert vals.getSize() == 4


def test_bind_keep_alive_temporary_view() -> None:
    """bind() keeps a temporary source view alive via keep_alive."""
    import gc  # noqa: PLC0415

    m = DenseMatrix[float]()
    m.setDimensions(2, 2)
    m.setValue(42.0)

    target = DenseMatrix[float]()
    target.setDimensions(2, 2)
    target_view = target.getView()

    # bind to a temporary view — without keep_alive, this temporary would be GC'd
    target_view.bind(m.getView())
    gc.collect()

    assert target_view.getElement(0, 0) == 42.0


# ---------------------------------------------------------------------------
# isBinary / isSymmetric (def_static on MatrixBase, inherited by owners & views)
# ---------------------------------------------------------------------------


def test_is_binary_is_symmetric() -> None:
    """isBinary/isSymmetric return False for GeneralMatrix on owner and view."""
    m = create_matrix(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])

    # Instance-level calls on the owning matrix
    assert m.isBinary() is False
    assert m.isSymmetric() is False

    # Static calls on the class itself
    assert DM.isBinary() is False
    assert DM.isSymmetric() is False

    # Instance-level calls on a mutable view
    view = m.getView()
    assert view.isBinary() is False
    assert view.isSymmetric() is False

    # Instance-level calls on a const view
    cview = m.getConstView()
    assert cview.isBinary() is False
    assert cview.isSymmetric() is False


# ---------------------------------------------------------------------------
# Organization parameter
# ---------------------------------------------------------------------------


def test_organization_subscript_returns_correct_class() -> None:
    """DenseMatrix subscript with organization resolves to correct C++ class."""
    assert DenseMatrix[float, Host, ElementsOrganization.RowMajorOrder] is DM
    assert DenseMatrix[float, Host, ElementsOrganization.ColumnMajorOrder] is pytnl._matrices.DenseMatrix_float_ColumnMajor


def test_organization_default() -> None:
    """DenseMatrix[float] defaults to RowMajor on Host."""
    assert DenseMatrix[float] is DenseMatrix[float, Host, ElementsOrganization.RowMajorOrder]


def test_getOrganization_returns_correct_enum() -> None:
    """getOrganization() returns the correct ElementsOrganization enum value."""
    m_row = DenseMatrix[float, Host, ElementsOrganization.RowMajorOrder]()
    assert m_row.getOrganization() == ElementsOrganization.RowMajorOrder

    m_col = DenseMatrix[float, Host, ElementsOrganization.ColumnMajorOrder]()
    assert m_col.getOrganization() == ElementsOrganization.ColumnMajorOrder


def test_organization_dlpack_strides() -> None:
    """DLPack strides match the matrix organization."""
    rows, cols = 3, 4
    m_row = DenseMatrix[float, Host, ElementsOrganization.RowMajorOrder](rows, cols)
    arr_row = np.from_dlpack(m_row)
    assert arr_row.flags["C_CONTIGUOUS"]
    assert not arr_row.flags["F_CONTIGUOUS"]

    m_col = DenseMatrix[float, Host, ElementsOrganization.ColumnMajorOrder](rows, cols)
    arr_col = np.from_dlpack(m_col)
    assert arr_col.flags["F_CONTIGUOUS"]
    assert not arr_col.flags["C_CONTIGUOUS"]


def test_organization_invalid_org_raises() -> None:
    """Invalid organization raises TypeError."""

    class BadOrg:
        pass

    with pytest.raises(TypeError):
        DenseMatrix[float, Host, BadOrg]  # type: ignore[index]


def test_view_organization_subscript_returns_correct_class() -> None:
    """DenseMatrixView subscript with organization resolves to correct C++ class."""
    assert DenseMatrixView[float, Host, ElementsOrganization.RowMajorOrder] is DV
    assert DenseMatrixView[float, Host, ElementsOrganization.ColumnMajorOrder] is pytnl._matrices.DenseMatrixView_float_ColumnMajor

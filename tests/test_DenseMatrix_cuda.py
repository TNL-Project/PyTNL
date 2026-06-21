# mypy: disable-error-code="attr-defined, no-any-unimported, unused-ignore, valid-type"
# pyright: standard
# pyright: reportMissingImports=information
# pyright: reportInvalidTypeForm=none

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pytnl.containers import Vector
from pytnl.devices import Cuda
from pytnl.matrices import DenseMatrix

if TYPE_CHECKING:
    import pytnl._containers_cuda as _containers_cuda  # type: ignore[import-not-found]
    import pytnl._matrices_cuda as _matrices_cuda  # type: ignore[import-not-found]
else:
    _containers_cuda = pytest.importorskip("pytnl._containers_cuda")
    _matrices_cuda = pytest.importorskip("pytnl._matrices_cuda")

cp = pytest.importorskip("cupy")

# Mark all tests in this module
pytestmark = pytest.mark.cuda

# ---------------------------------------------------------------------------
# Single CUDA matrix type (no format parameter — only value_type + device_type)
# ---------------------------------------------------------------------------

DM = _matrices_cuda.DenseMatrix_float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_matrix(
    rows: int,
    cols: int,
    entries: list[tuple[int, int, float]],
) -> DM:
    """Create a dense matrix filled with the given entries.

    ``entries`` is a list of ``(row, col, value)`` tuples.
    All elements default to 0.0; only the listed entries are set.
    No ``setRowCapacities`` call is needed for dense matrices.
    """
    m = DenseMatrix[float, Cuda]()
    m.setDimensions(rows, cols)
    m.setValue(0.0)
    for r, c, v in entries:
        m.setElement(r, c, v)
    return m  # type: ignore[no-any-return]


def identity_matrix(size: int) -> DM:
    """Create a size x size identity matrix."""
    entries = [(i, i, 1.0) for i in range(size)]
    return create_matrix(size, size, entries)


# ---------------------------------------------------------------------------
# Subscript / pythonization: correct class resolution
# ---------------------------------------------------------------------------


def test_subscript_cuda() -> None:
    """Verify subscript syntax resolves to the correct CUDA C++ class."""
    assert DenseMatrix[float, Cuda] is DM


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
    """setElement / getElement round-trip; unset elements are 0.0."""
    m = create_matrix(2, 3, [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)])

    assert m.getElement(0, 0) == 1.0
    assert m.getElement(0, 2) == 2.5
    assert m.getElement(1, 1) == -3.0
    # Unset elements default to 0.0 (dense stores all elements)
    assert m.getElement(0, 1) == 0.0
    assert m.getElement(1, 0) == 0.0
    assert m.getElement(1, 2) == 0.0


def test_addElement() -> None:
    """addElement accumulates values at the same position."""
    m = create_matrix(1, 2, [(0, 0, 1.0), (0, 1, 2.0)])
    m.addElement(0, 0, 10.0, 1.0)
    m.addElement(0, 0, 20.0, 1.0)
    assert m.getElement(0, 0) == 31.0  # 1.0 + 10.0 + 20.0


def test_setValue_all_elements() -> None:
    """setValue sets ALL elements to the given value (DenseMatrix-specific)."""
    m = DM(2, 3)
    m.setValue(5.0)
    for r in range(2):
        for c in range(3):
            assert m.getElement(r, c) == 5.0


def test_reset() -> None:
    """reset() must zero out dimensions."""
    m = create_matrix(2, 2, [(0, 0, 1.0)])
    assert m.getRows() == 2
    m.reset()
    assert m.getRows() == 0
    assert m.getColumns() == 0


# ---------------------------------------------------------------------------
# RowView (skipped on CUDA — getValue returns device memory reference)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# vectorProduct
# ---------------------------------------------------------------------------


def test_vectorProduct() -> None:
    """vectorProduct of identity matrix x vector equals the input vector."""
    n = 3
    m = identity_matrix(n)

    in_vec = Vector[float, Cuda](n)
    for i in range(n):
        in_vec[i] = float(i + 1)

    out_vec = Vector[float, Cuda](n)
    m.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)

    for i in range(n):
        assert out_vec[i] == pytest.approx(float(i + 1))


def test_vectorProduct_with_factors() -> None:
    """vectorProduct with matrixMult=2.0, outMult=3.0 on identity matrix."""
    n = 3
    m = identity_matrix(n)

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


def test_equality() -> None:
    """Two matrices with identical content are equal by ==."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0)]
    m1 = create_matrix(2, 3, entries)
    m2 = create_matrix(2, 3, entries)
    assert m1 == m2  # type: ignore[operator]


def test_inequality() -> None:
    """Matrices with different content must not be equal."""
    m1 = create_matrix(2, 3, [(0, 0, 1.0), (0, 2, 2.5)])
    m2 = create_matrix(2, 3, [(0, 0, 1.0), (0, 2, 9.9)])
    assert m1 != m2  # type: ignore[operator]

    # Different number of non-zero elements
    m3 = create_matrix(2, 3, [(0, 0, 1.0)])
    m4 = create_matrix(2, 3, [(0, 0, 1.0), (1, 1, 2.0)])
    assert m3 != m4  # type: ignore[operator]


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
    """getRowCapacities / getRowCapacity / getCompressedRowLengths for dense."""
    entries = [(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (2, 0, 4.0), (2, 1, 5.0), (2, 2, 6.0)]
    m = create_matrix(3, 4, entries)

    # getRowCapacities writes into a pre-allocated vector
    result = _containers_cuda.Vector_int(3)
    m.getRowCapacities(result)
    assert result.getSize() == 3
    # For dense, row capacity = number of columns
    assert result[0] == 4
    assert result[1] == 4
    assert result[2] == 4

    # getRowCapacity for individual rows
    assert m.getRowCapacity(0) == 4
    assert m.getRowCapacity(1) == 4
    assert m.getRowCapacity(2) == 4

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


def test_nonzero_counts() -> None:
    """getNonzeroElementsCount matches entry count; allocated = rows * cols."""
    entries = [(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 1, 5.0)]
    m = create_matrix(3, 3, entries)

    assert m.getNonzeroElementsCount() == 5
    # For dense, allocated = rows * cols (all elements are allocated)
    assert m.getAllocatedElementsCount() == 9


def test_allocated_elements_count() -> None:
    """getAllocatedElementsCount returns rows * cols for dense matrices."""
    m = DM(3, 4)
    assert m.getAllocatedElementsCount() == 12


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


# ---------------------------------------------------------------------------
# String representation
# ---------------------------------------------------------------------------


def test_str_output() -> None:
    """str(matrix) returns non-empty string containing row indicators."""
    m = create_matrix(2, 3, [(0, 0, 1.0), (1, 2, 3.0)])
    s = str(m)
    assert isinstance(s, str)
    assert len(s) > 0
    # Should contain row indicators
    assert "0:" in s or "Row" in s or "[" in s


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
@settings(deadline=None)
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
@settings(deadline=None)
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

    in_vec = Vector[float, Cuda](n)
    for i, val in enumerate(in_values):
        in_vec[i] = val

    out_vec = Vector[float, Cuda](n)
    m.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)

    expected = dense @ np.array(in_values)
    for i in range(n):
        assert out_vec[i] == pytest.approx(expected[i], rel=1e-10)


# ---------------------------------------------------------------------------
# DLPack interoperability (CUDA → CuPy)
# ---------------------------------------------------------------------------


def test_dlpack_device_cuda() -> None:
    """__dlpack_device__ returns CUDA device for Cuda matrices."""
    m = DM(2, 3)
    device, _ = m.__dlpack_device__()
    assert device == 2


def test_dlpack_to_cupy_shape_dtype() -> None:
    """DLPack export produces a cupy array with correct shape and dtype."""
    m = create_matrix(3, 4, [(0, 0, 1.0), (1, 2, 3.5), (2, 3, -2.0)])
    arr = cp.from_dlpack(m)
    assert arr.shape == (3, 4)
    assert arr.dtype == cp.float64


def test_dlpack_to_cupy_values() -> None:
    """DLPack export preserves matrix element values on CUDA."""
    entries = [(0, 0, 1.0), (0, 2, 2.5), (1, 1, -3.0), (2, 3, 4.0)]
    m = create_matrix(3, 4, entries)
    arr = cp.from_dlpack(m)

    for r, c, v in entries:
        assert float(arr[r, c]) == v
    assert float(arr[0, 1]) == 0.0


def test_dlpack_zero_copy_cuda() -> None:
    """DLPack export is zero-copy — modifying the cupy array modifies the matrix."""
    m = create_matrix(2, 2, [(0, 0, 1.0), (1, 1, 2.0)])
    arr = cp.from_dlpack(m)
    arr[0, 1] = 99.0
    assert m.getElement(0, 1) == 99.0


def test_dlpack_strides_columnmajor() -> None:
    """CUDA DenseMatrix (ColumnMajor) has Fortran-contiguous strides."""
    m = DM(3, 4)
    arr = cp.from_dlpack(m)
    assert arr.flags["F_CONTIGUOUS"]

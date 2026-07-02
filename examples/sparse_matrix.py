# mypy: disable-error-code="type-arg, union-attr"

import random
import tempfile
from importlib.util import find_spec
from pathlib import Path

from pytnl._containers import Vector_int
from pytnl.containers import Vector
from pytnl.devices import Cuda, Host
from pytnl.matrices import SparseMatrix, copySparseMatrix, formats

SIZE = 100

CSR = SparseMatrix[float, Host, formats.CSR]
Ellpack = SparseMatrix[float, Host, formats.Ellpack]
SlicedEllpack = SparseMatrix[float, Host, formats.SlicedEllpack]

SparseMatrixType = CSR | Ellpack | SlicedEllpack


def create_matrix(matrix: SparseMatrixType, size: int, capacities: Vector_int) -> SparseMatrixType:
    matrix.setDimensions(size, size)
    matrix.setRowCapacities(capacities)
    return matrix


def fill_random(matrix: SparseMatrixType, p: float = 0.1) -> None:
    for i in range(matrix.getRows()):
        for j in range(matrix.getColumns()):
            if random.random() < p:
                matrix.addElement(i, j, random.random(), 1)


def set_elements(matrix: SparseMatrixType, rows: int, cols: int, entries: list[tuple[int, int, float]]) -> None:
    matrix.setDimensions(rows, cols)
    capacities = Vector[int](rows)
    for i in range(rows):
        capacities[i] = 0
    for r, _c, _v in entries:
        capacities[r] += 1
    matrix.setRowCapacities(capacities)
    for r, c, v in entries:
        matrix.setElement(r, c, v)


def print_matrix(name: str, matrix: SparseMatrixType) -> None:
    print(f"{name} matrix:")
    print(matrix)
    print(f"rows: {matrix.getRows()}, cols: {matrix.getColumns()}, nnz: {matrix.getNonzeroElementsCount()}, allocated: {matrix.getAllocatedElementsCount()}")
    print()


def save_matrix(matrix: SparseMatrixType, name: str) -> SparseMatrixType:
    matrix.save(name)
    return matrix


def load_matrix(matrix: SparseMatrixType, name: str) -> SparseMatrixType:
    matrix.load(name)
    return matrix


def demo_coordinate_construction() -> None:
    """Build a CSR matrix from explicit (row, col, value) tuples.

    Demonstrates computing per-row capacities from the coordinate list and
    using ``setElement`` (which overwrites) instead of ``addElement``.
    """
    rows = 5
    cols = 5
    nnz = 8

    print("--- Coordinate-based CSR construction ---")
    print(f"Dimensions: {rows}x{cols}, non-zero elements: {nnz}")

    random.seed(42)

    coordinates: list[tuple[int, int, float]] = []
    already_filled: set[tuple[int, int]] = set()
    while len(coordinates) < nnz:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        v = random.random()
        if (r, c) not in already_filled:
            coordinates.append((r, c, v))
            already_filled.add((r, c))

    csr = CSR()
    csr.setDimensions(rows, cols)

    row_capacities = Vector[int](rows, 0)
    for r, _c, _v in coordinates:
        row_capacities[r] += 1
    csr.setRowCapacities(row_capacities)

    for r, c, v in coordinates:
        csr.setElement(r, c, v)

    print("\n[CSR Matrix]")
    print(csr)
    print()


def demo_save_load_equality() -> None:
    random.seed(42)

    row_capacities = Vector[int](SIZE, SIZE)

    matrices: list[tuple[str, SparseMatrixType]] = [
        ("CSR", CSR()),
        ("CSR1", CSR()),
        ("Ell", Ellpack()),
        ("SEll", SlicedEllpack()),
    ]

    loaded_matrices: list[tuple[str, SparseMatrixType]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, m in matrices:
            filename = str(Path(tmpdir) / f"mat{name}")
            mat = create_matrix(m, SIZE, row_capacities)
            fill_random(mat)
            save_matrix(mat, filename)
            lmat = load_matrix(mat, filename)
            loaded_matrices.append((name, lmat))

    for name, mat in loaded_matrices:
        print_matrix(name, mat)

    csr = loaded_matrices[0][1]
    csr1 = loaded_matrices[1][1]

    equal = csr == csr1
    nequal = csr != csr1
    print(f"equal: {equal}, not equal: {nequal}")


def demo_copy_sparse_matrix() -> None:
    """Convert a CSR matrix to Ellpack and back via copySparseMatrix."""
    entries = [
        (0, 0, 1.0),
        (0, 2, 2.5),
        (1, 1, -3.0),
        (2, 0, 4.0),
        (2, 4, 7.0),
        (3, 3, 0.5),
        (4, 1, 9.0),
    ]
    rows = 5
    cols = 5

    print("--- copySparseMatrix on Host ---\n")

    csr = CSR()
    set_elements(csr, rows, cols, entries)

    ell = Ellpack()
    ell.setDimensions(rows, cols)
    copySparseMatrix(ell, csr)
    print(f"CSR -> Ellpack: nnz={ell.getNonzeroElementsCount()}")
    for r, c, v in entries:
        assert ell.getElement(r, c) == v

    csr2 = CSR()
    csr2.setDimensions(rows, cols)
    copySparseMatrix(csr2, ell)
    print(f"Ellpack -> CSR: nnz={csr2.getNonzeroElementsCount()}")
    for r, c, v in entries:
        assert csr2.getElement(r, c) == v

    if find_spec("pytnl._matrices_cuda") is None:
        print("\n(CUDA module not available — skipping Cuda)")
        return

    print("\n--- copySparseMatrix cross-device ---\n")

    cuda_ell = SparseMatrix[float, Cuda, formats.Ellpack]()
    cuda_ell.setDimensions(rows, cols)
    copySparseMatrix(cuda_ell, csr)
    print(f"Host CSR -> Cuda Ellpack: nnz={cuda_ell.getNonzeroElementsCount()}")
    for r, c, v in entries:
        assert cuda_ell.getElement(r, c) == v

    host_csr = CSR()
    host_csr.setDimensions(rows, cols)
    copySparseMatrix(host_csr, cuda_ell)
    print(f"Cuda Ellpack -> Host CSR: nnz={host_csr.getNonzeroElementsCount()}")
    for r, c, v in entries:
        assert host_csr.getElement(r, c) == v

    print()


def main() -> None:
    demo_coordinate_construction()
    demo_save_load_equality()
    demo_copy_sparse_matrix()


if __name__ == "__main__":
    main()

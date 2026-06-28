# mypy: disable-error-code="type-arg, union-attr"

import random
import tempfile
from pathlib import Path

from pytnl._containers import Vector_int
from pytnl.containers import Vector
from pytnl.devices import Host
from pytnl.matrices import SparseMatrix, formats

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


def main() -> None:
    demo_coordinate_construction()
    demo_save_load_equality()


if __name__ == "__main__":
    main()

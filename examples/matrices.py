# mypy: disable-error-code="type-arg, union-attr"
# pyright: reportAttributeAccessIssue=none, reportInvalidTypeForm=none, reportUnknownArgumentType=none, reportUnknownMemberType=none, reportUnknownParameterType=none, reportUnknownVariableType=none

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


def fill_random(matrix: SparseMatrixType, size: int, p: float = 0.1) -> None:
    for i in range(size):
        for j in range(size):
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


def main() -> None:
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
            fill_random(mat, SIZE)
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


if __name__ == "__main__":
    main()

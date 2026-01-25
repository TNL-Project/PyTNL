import random
import sys
from datetime import UTC, datetime

from pytnl._containers import Vector_int
from pytnl.matrices import CSR, Ellpack, SlicedEllpack

SIZE = 100
VECTOR = Vector_int(SIZE, SIZE)

MatrixType = CSR | Ellpack | SlicedEllpack


def createMatrix(matrix: MatrixType, size: int, vector: Vector_int) -> MatrixType:
    matrix.setDimensions(size, size)
    matrix.setRowCapacities(vector)
    return matrix


def fillRandom(matrix: MatrixType, size: int, p: float = 0.1) -> None:
    for i in range(size):
        for j in range(size):
            if random.random() < p:
                matrix.addElement(i, j, random.random(), 1)


def printMatrix(name: str, matrix: MatrixType) -> None:
    print(f"{name} matrix:")
    print(matrix)
    print(
        "rows:",
        matrix.getRows(),
        "cols:",
        matrix.getColumns(),
        "nnz:",
        matrix.getNonzeroElementsCount(),
        "allocated:",
        matrix.getAllocatedElementsCount(),
        "memory use:",
        sys.getsizeof(matrix),
        #   "serialization:", matrix.getSerializationType(),
    )
    print()


def getDate() -> str:
    curr_date = datetime.now(UTC).date()
    str_date = str(curr_date)
    return str_date


def saveMatrix(matrix: MatrixType, name: str) -> MatrixType:
    matrix.save(name)
    return matrix


def loadMatrix(matrix: MatrixType, name: str) -> MatrixType:
    matrix.load(name)
    return matrix


current_date = getDate()

matrices = [
    ("CSR", CSR()),
    ("CSR1", CSR()),
    ("Ell", Ellpack()),
    ("SEll", SlicedEllpack())
]

loaded_matrices = []

for name, m in matrices:
    mat = createMatrix(m, SIZE, VECTOR)
    fillRandom(mat, SIZE)
    saveMatrix(mat, f"mat{name}_{current_date}")
    lmat = loadMatrix(mat, f"mat{name}_{current_date}")
    loaded_matrices.append((name, lmat))

for name, mat in loaded_matrices:
    printMatrix(name, mat)


csr = loaded_matrices[0][1]  # csr
csr1 = loaded_matrices[1][1]  # csr1

equal = csr.__eq__(csr1)
nequal = csr.__ne__(csr1)
print(equal, ", ", nequal)

import random
import sys
from datetime import date

from pytnl._containers import Vector_int
from pytnl.matrices import CSR, Ellpack, SlicedEllpack

SIZE = 1000
VECTOR = Vector_int(SIZE, SIZE)

def createMatrix(matrix, size, vector):
    m = matrix
    m.setDimensions(size, size)
    m.setRowCapacities(vector)
    return matrix

def fillRandom(matrix, size, p=0.1) -> None:
    for i in range(size):
        for j in range(size):
            if random.random() < p:
                matrix.addElement(i, j, random.random(), 1)

def printMatrix(name, matrix) -> None:
    print(f"{name} matrix:")
    # print(matrix)
    print("rows:", matrix.getRows(),
          "cols:", matrix.getColumns(),
          "nnz:", matrix.getNonzeroElementsCount(),
          "allocated:", matrix.getAllocatedElementsCount(),
          "serialization:", matrix.getSerializationType(),
          "memory use:", sys.getsizeof(matrix))
    print()

def getDate() -> str:
    curr_date = date.today()
    str_date = str(curr_date)
    return str_date

def saveMatrix(matrix, name):
    matrix.save(name)

def loadMatrix(matrix, name):
    loaded_matrix = matrix
    loaded_matrix.load(name)
    return loaded_matrix

current_date = getDate()

csr = createMatrix(CSR(), SIZE, VECTOR)
csr1 = createMatrix(CSR(), SIZE, VECTOR)
ellpack = createMatrix(Ellpack(), SIZE, VECTOR)
sellpack = createMatrix(SlicedEllpack(), SIZE, VECTOR)

fillRandom(csr, SIZE)
fillRandom(csr1, SIZE)
fillRandom(ellpack, SIZE)
fillRandom(sellpack, SIZE)

csr_saved = saveMatrix(csr, "matCSR_" + current_date)
ellpack_saved = saveMatrix(ellpack, "matEllpack_" + current_date)
sellpack_saved = saveMatrix(sellpack, "matSellpack_" + current_date)

csr_loaded = loadMatrix(csr, "matCSR_" + current_date)
ellpack_loaded = loadMatrix(ellpack, "matEllpack_" + current_date)
sellpack_loaded = loadMatrix(sellpack, "matSellpack_" + current_date)

# printMatrix("CSR", csr_loaded)
# printMatrix("Ellpack", ellpack_loaded)
# printMatrix("Sliced Ellpack", sellpack_loaded)

equal = csr.__eq__(csr1)
print(equal)

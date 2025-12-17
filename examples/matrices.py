from pytnl.matrices import CSR, Ellpack, SparseMatrixRowView
from pytnl._containers import Vector_int
import numpy as np
import time
import random

# TODO
# 1. addition
# 2. multiplication
# 3. inverse
# 4. norm
# 5. load
# 6. save
# 7. reset

# region pytnl

# create a compressed sparse row matrix
tnl_matrix = CSR()
tnl_vector = Vector_int(10, 10)
tnl_matrix.setDimensions(10, 10)
tnl_matrix.setRowCapacities(tnl_vector)
# tnl_matrix.addElement(3, 5, 5.2, 1)

# fill the matrix 
val = random.uniform(0, 2)

for i in range(10):
    for j in range(10):
        if random.random() < 0.1:
            val = random.random()
            tnl_matrix.addElement(i, j, val, 1)

# rows, cols = tnl_matrix.getRows(), tnl_matrix.getColumns()
# dense = np.zeros((rows, cols))

# for r in range(rows):
#     row_data = tnl_matrix.getRow(r)
#     for col, val in row_data:
#         dense[r, col] = val

columns = tnl_matrix.getColumns()
rows = tnl_matrix.getRows()

print("columns:", columns, "rows:", rows, "matrix:", tnl_matrix)

# endregion


# region numpy

b_matrix = np.array()
b_matrix([1, 0, 0], [0, 1, 0], [0, 0, 1])


# endregion







# region scipy

# endregion







# region python

# endregion

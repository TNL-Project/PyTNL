from pytnl.matrices import CSR, Ellpack, SparseMatrixRowView
from pytnl._containers import Vector_int
import numpy as np
# import scipy as sp
import sys
import time


# TODO
# 1. addition
# 2. multiplication
# 3. inverse
# 4. norm
# 5.

# region pytnl

a_matrix = CSR()
a_vector = Vector_int(10, 50)
a_matrix.setDimensions(10, 10)
a_matrix.setRowCapacities(a_vector)

a_matrix.addElement(3, 5, 5.2)
# a_matrix.print()


columns = a_matrix.getColumns()
rows = a_matrix.getRows()

print("columns:", columns, "rows:", rows, "matrix:", a_matrix)

# endregion

# region numpy

# b_matrix = np.array()
# b_matrix([1, 0, 0], [0, 1, 0], [0, 0, 1])


# endregion







# region scipy

# endregion







# region python

# endregion

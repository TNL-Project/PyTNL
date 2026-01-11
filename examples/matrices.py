import random
from datetime import date

from pytnl._containers import Vector_int
from pytnl.matrices import CSR, Ellpack, SlicedEllpack

SIZE = 10

# TODO
# 1. addition
# 2. multiplication
# 3. inverse
# 4. norm
# 5. load
# 6. save
# 7. reset

# region CSR

# create a compressed sparse row matrix
csr_matrix = CSR()
tnl_vector = Vector_int(SIZE, SIZE)
csr_matrix.setDimensions(SIZE, SIZE)
csr_matrix.setRowCapacities(tnl_vector)

# fill the matrix
val = random.uniform(0, 5)

for i in range(SIZE):
    for j in range(SIZE):
        if random.random() < 0.1:
            val = random.random()
            csr_matrix.addElement(i, j, val, 1)

columns = csr_matrix.getColumns()
rows = csr_matrix.getRows()

# print the matrix
print("columns:", columns, "rows:", rows, "matrix:", csr_matrix)

#get the current date for later use
date = date.today()
str_date = str(date)

# save the matrix into a file
csr_matrix.save(str_date)

# load the matrix from a file
loaded_matrix = csr_matrix.load(str_date)
print(loaded_matrix)

# endregion

# region Ellpack
# endregion



# region SlicedEllpack
# endregion

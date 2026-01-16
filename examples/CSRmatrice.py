# pyright: reportAttributeAccessIssue=none
# pyright: reportUnknownArgumentType=none
# pyright: reportUnknownMemberType=none
# pyright: reportUnknownVariableType=none

# Import for memory handling

import random

from pytnl.containers import Vector
from pytnl.matrices import CSR


def main() -> None:
    print("--- CSR Matrix Creator ---")

    # 1. Ask for dimensions with security checks
    try:
        rows = int(input("Number of rows: "))
        cols = int(input("Number of columns: "))
        nnz = int(input("Number of non-zero elements: "))
    except ValueError:
        print("Error: Dimensions must be integers. Exiting.")
        return
    if nnz > rows * cols:
        print(f"Error: Too many non-zero elements ({nnz}) for the matrix size ({rows}x{cols}). Exiting.")
        return

    # 2. Generating random values to fill matrice
    numbers: list[tuple[int, int, float]] = []
    already_filled: list[tuple[int, int]] = []
    while len(numbers) < nnz:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        v = random.random()
        if (r, c) not in already_filled:
            numbers.append((r, c, v))
            already_filled.append((r, c))

    # 3. Build CSR Matrix
    csr = CSR()
    csr.setDimensions(rows, cols)

    # Calculate capacities per row
    caps = Vector[int]()
    caps.setSize(rows)
    for i in range(rows):
        caps[i] = 0

    for r, c, v in numbers:
        caps[r] += 1

    csr.setRowCapacities(caps)

    # Fill the matrix
    for r, c, v in numbers:
        csr.setElement(r, c, v)

    print("\n[Original Matrix]")
    print(csr)


if __name__ == "__main__":
    main()

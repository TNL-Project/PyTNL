# pyright: reportAttributeAccessIssue=none, reportUnknownArgumentType=none, reportUnknownMemberType=none, reportUnknownVariableType=none

import random

from pytnl.containers import Vector
from pytnl.matrices import SparseMatrix

ROWS = 5
COLS = 5
NNZ = 8


def main() -> None:
    print("--- CSR Matrix Creator ---")
    print(f"Dimensions: {ROWS}x{COLS}, non-zero elements: {NNZ}")

    random.seed(42)

    numbers: list[tuple[int, int, float]] = []
    already_filled: set[tuple[int, int]] = set()
    while len(numbers) < NNZ:
        r = random.randint(0, ROWS - 1)
        c = random.randint(0, COLS - 1)
        v = random.random()
        if (r, c) not in already_filled:
            numbers.append((r, c, v))
            already_filled.add((r, c))

    csr = SparseMatrix[float]()
    csr.setDimensions(ROWS, COLS)

    caps = Vector[int](ROWS, 0)

    for r, c, v in numbers:
        caps[r] += 1

    csr.setRowCapacities(caps)

    for r, c, v in numbers:
        csr.setElement(r, c, v)

    print("\n[CSR Matrix]")
    print(csr)


if __name__ == "__main__":
    main()

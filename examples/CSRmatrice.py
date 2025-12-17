import pytnl
from pytnl.matrices import CSR
# Import for memory handling
try:
    from pytnl.containers import Vector_int
except ImportError:
    from pytnl._containers import Vector_int

def main():
    print("--- CSR Matrix Creator ---")

    # 1. Ask for dimensions
    rows = int(input("Number of rows: "))
    cols = int(input("Number of columns: "))
    nnz = int(input("Number of non-zero elements: "))

    # 2. Collect data
    entries = []
    for i in range(nnz):
        raw = input(f"Element {i+1} (row col value): ").split()
        entries.append((int(raw[0]), int(raw[1]), float(raw[2])))

    # 3. Build CSR Matrix
    csr = CSR()
    csr.setDimensions(rows, cols)

    # Calculate capacities per row
    caps = Vector_int(rows)
    for i in range(rows): caps[i] = 0

    for r, c, v in entries:
        caps[r] += 1

    csr.setRowCapacities(caps)

    # Fill the matrix
    for r, c, v in entries:
        csr.setElement(r, c, v)

    print("\n[Original Matrix]")
    print(csr)

if __name__ == "__main__":
    main()

import pytnl
from pytnl.matrices import CSR
# Import for memory handling


def main():
    print("--- CSR Matrix Creator ---")

    # 1. Ask for dimensions with security checks
    try:
        rows = int(input("Number of rows: "))
        cols = int(input("Number of columns: "))
        nnz = int(input("Number of non-zero elements: "))
    except ValueError:
        print("Error: Dimensions must be integers. Exiting.")
        return

    #entries with security checks
    entries = []
    for i in range(nnz):
        try:
            line = input(f"Element {i+1} (row col value): ").split()

            # Format check
            if len(line) < 3:
                print("   [!] Ignored: Invalid format. Usage: row col value")
                continue

            r = int(line[0])
            c = int(line[1])
            v = float(line[2])

            # avoid C from crashing on invalid indices
            if r < 0 or r >= rows:
                print(f"   [!] Ignored: Row index {r} is out of bounds (0-{rows-1}).")
                continue
            if c < 0 or c >= cols:
                print(f"   [!] Ignored: Column index {c} is out of bounds (0-{cols-1}).")
                continue

            # If ok, add to list
            entries.append((r, c, v))

        except ValueError:
            print("   [!] Ignored: Inputs must be numbers.")

    if not entries:
        print("No valid data provided. Exiting.")
        return

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

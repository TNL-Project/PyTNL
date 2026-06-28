from pytnl.containers import Vector
from pytnl.devices import Host
from pytnl.matrices import SparseMatrix, SparseMatrixRowView, formats

ROWS = 5
COLS = 5


def print_row(row: SparseMatrixRowView) -> None:
    """Print the stored entries of a sparse row view."""
    print(f"  Row {row.getRowIndex()} has {row.getSize()} allocated slot(s):")
    for i in range(row.getSize()):
        col = row.getColumnIndex(i)
        val = row.getValue(i)
        print(f"    slot {i}: column {col}, value {val}")


def main() -> None:
    print("--- SparseMatrixView Example ---")
    print(f"Dimensions: {ROWS}x{COLS}\n")

    entries = [
        (0, 0, 1.0),
        (1, 2, 3.5),
        (2, 4, -2.0),
        (4, 4, 5.5),
    ]

    matrix = SparseMatrix[float, Host, formats.CSR]()
    matrix.setDimensions(ROWS, COLS)

    capacities = Vector[int](ROWS, 0)
    for row, _, _ in entries:
        capacities[row] += 1
    matrix.setRowCapacities(capacities)

    for row, col, value in entries:
        matrix.setElement(row, col, value)

    view = matrix.getView()

    print("[Parent matrix]")
    print(matrix)
    print("\n[Mutable view]")
    print(view)

    print("\nAccessing elements through the view:")
    print(f"  view.getElement(0, 0) = {view.getElement(0, 0)}")
    print(f"  view.getElement(1, 2) = {view.getElement(1, 2)}")
    print(f"  view.getElement(2, 4) = {view.getElement(2, 4)}")

    print("\nModifying elements through the view:")
    view.setElement(1, 2, 10.0)
    print("  view.setElement(1, 2, 10.0)")
    print(f"  matrix.getElement(1, 2) = {matrix.getElement(1, 2)}")

    const_view = matrix.getConstView()
    print("\n[Const view]")
    print(const_view)

    print("\nRow access through the view:")
    print_row(view.getRow(0))

    print("\nVector product through the mutable view:")
    in_vec = Vector[float](COLS, 1.0)
    out_vec = Vector[float](ROWS, 0.0)
    view.vectorProduct(in_vec, out_vec, 1.0, 0.0, 0, 0)
    print(f"  Input vector:  {[in_vec[i] for i in range(COLS)]}")
    print(f"  Output vector: {[out_vec[i] for i in range(ROWS)]}")

    print("\nBounds checking (out-of-bounds access raises IndexError):")
    try:
        view.getElement(ROWS, 0)
    except IndexError as exc:
        print(f"  Caught expected error: {exc}")


if __name__ == "__main__":
    main()

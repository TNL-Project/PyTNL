import tempfile
from pathlib import Path

import numpy as np

from pytnl.containers import Vector
from pytnl.matrices import DenseMatrix

ROWS = 5
COLS = 5


def main() -> None:
    print("--- DenseMatrix Example ---")
    print(f"Dimensions: {ROWS}x{COLS}")

    # Create a dense matrix
    m = DenseMatrix[float]()
    m.setDimensions(ROWS, COLS)

    # Set all elements to zero
    m.setValue(0.0)

    # Fill with some values
    m.setElement(0, 0, 1.0)
    m.setElement(1, 2, 3.5)
    m.setElement(2, 4, -2.0)
    m.setElement(4, 4, 5.5)

    print("\n[Dense Matrix]")
    print(m)

    # Get element values
    val_00 = m.getElement(0, 0)
    val_12 = m.getElement(1, 2)
    val_33 = m.getElement(3, 3)
    print(f"\nElement (0,0): {val_00}")
    print(f"Element (1,2): {val_12}")
    print(f"Element (3,3): {val_33}")

    # Get dimensions
    rows = m.getRows()
    cols = m.getColumns()
    print(f"\nDimensions: {rows}x{cols}")

    # Access a row using RowView
    row = m.getRow(0)
    row_size = row.getSize()
    row_index = row.getRowIndex()
    row_val_0 = row.getValue(0)
    row_val_1 = row.getValue(1)
    print("\nRow 0 view:")
    print(f"  Row index: {row_index}")
    print(f"  Size (columns): {row_size}")
    print(f"  Value at column 0: {row_val_0}")
    print(f"  Value at column 1: {row_val_1}")

    # Compute vector product: out = alpha * m * in + beta * out
    in_vec = Vector[float](COLS, 1.0)
    out_vec = Vector[float](ROWS, 0.0)
    alpha = 1.0
    beta = 0.0
    m.vectorProduct(in_vec, out_vec, alpha, beta, 0, 0)

    print("\nVector product result (m * [1,1,1,1,1]):")
    print(f"  Output vector: {[out_vec[i] for i in range(ROWS)]}")

    # Compare two matrices
    m2 = DenseMatrix[float]()
    m2.setDimensions(ROWS, COLS)
    m2.setValue(0.0)
    m2.setElement(0, 0, 1.0)
    m2.setElement(1, 2, 3.5)
    m2.setElement(2, 4, -2.0)
    m2.setElement(4, 4, 5.5)

    equal = m == m2
    nequal = m != m2
    print("\nMatrix comparison:")
    print(f"  m == m2: {equal}")
    print(f"  m != m2: {nequal}")

    # Save and load round-trip
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = str(Path(tmpdir) / "dense_matrix.bin")

        print(f"\nSaving matrix to: {filename}")
        m.save(filename)

        m3 = DenseMatrix[float]()
        print(f"Loading matrix from: {filename}")
        m3.load(filename)

        restored_equal = m == m3
        print(f"Original == Restored: {restored_equal}")
        print("\n[Restored Matrix]")
        print(m3)

    print("\nDLPack interop:")
    device, _ = m.__dlpack_device__()
    print(f"  Device: {'CPU' if device == 1 else 'CUDA'}")

    arr = np.from_dlpack(m)
    print(f"  numpy shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  C-contiguous: {arr.flags['C_CONTIGUOUS']}")

    print("  numpy array (first 3 rows):")
    for r in range(min(3, ROWS)):
        print(f"    {arr[r].tolist()}")

    arr[0, 1] = 42.0
    print(f"\n  After arr[0,1] = 42.0, m.getElement(0,1) = {m.getElement(0, 1)}")


if __name__ == "__main__":
    main()

import numpy as np

from pytnl.matrices import DenseMatrix

ROWS = 4
COLS = 4


def main() -> None:
    print("--- DenseMatrixView Example ---")
    print(f"Dimensions: {ROWS}x{COLS}\n")

    matrix = DenseMatrix[float]()
    matrix.setDimensions(ROWS, COLS)
    matrix.setValue(0.0)

    entries = [
        (0, 0, 1.0),
        (1, 2, 3.5),
        (2, 3, -2.0),
        (3, 3, 5.5),
    ]
    for row, col, value in entries:
        matrix.setElement(row, col, value)

    view = matrix.getView()

    print("[Parent matrix]")
    print(matrix)
    print("\n[Mutable view]")
    print(view)

    print("\nDLPack export to NumPy (zero-copy):")
    arr = np.from_dlpack(view)
    print(f"  numpy shape: {arr.shape}, dtype: {arr.dtype}")
    print(f"  C-contiguous: {arr.flags['C_CONTIGUOUS']}")

    arr[0, 1] = 42.0
    print("  After arr[0, 1] = 42.0:")
    print(f"    view.getElement(0, 1) = {view.getElement(0, 1)}")
    print(f"    matrix.getElement(0, 1) = {matrix.getElement(0, 1)}")

    const_view = matrix.getConstView()
    print("\n[Const view]")
    print(const_view)

    print("\nConst view provides read-only access:")
    print(f"  const_view.getElement(0, 1) = {const_view.getElement(0, 1)}")
    print("  (DLPack export is available on the mutable view shown above.)")

    print("\nSetting all values through the view:")
    view.setValue(7.0)
    print(matrix)

    print("\nElement access and modification through the view:")
    view.setElement(2, 2, 9.0)
    print("  view.setElement(2, 2, 9.0)")
    print(f"  view.getElement(2, 2) = {view.getElement(2, 2)}")
    print(f"  matrix.getElement(2, 2) = {matrix.getElement(2, 2)}")

    print("\nBounds checking (out-of-bounds access raises IndexError):")
    try:
        view.getElement(ROWS, 0)
    except IndexError as exc:
        print(f"  Caught expected error: {exc}")


if __name__ == "__main__":
    main()

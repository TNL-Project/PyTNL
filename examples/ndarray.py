import numpy as np

from pytnl.containers import NDArray


def main() -> None:
    # Create a 2D array of floats
    a = NDArray[2, float]()
    a.setSizes(3, 4)

    # Initialize the array elements
    shape = a.getSizes()
    for i in range(shape[0]):
        for j in range(shape[1]):
            a[i, j] = i + j

    # Evaluate a function for all indices of the array
    def print_element(i: int, j: int) -> None:
        print(f"[{i}, {j}]: {a[i, j] = }")

    a.forAll(print_element)

    # Print the memory layout
    print(list(a.getStorageArrayView()))

    # NumPy interoperability via DLPack
    np_array = np.from_dlpack(a)
    print(np_array)
    print(type(np_array), np_array.shape, np_array.dtype)


if __name__ == "__main__":
    main()

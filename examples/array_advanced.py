import numpy as np

from pytnl.containers import Array


def main() -> None:
    # 1. Initialization and Element-wise Modification
    arr = Array[int](10, 0)

    for i in range(5):
        arr[i] = i * 2

    print("Array after initialization and filling indices 0-4:")
    print(str(arr))

    # 2. Slice Assignment
    data_list: list[int] = [10, 20, 30, 40, 50]
    array_temp = Array[int](len(data_list))
    for i, val in enumerate(data_list):
        array_temp[i] = val

    arr[5:10] = array_temp

    print("\nArray after slice assignment [5:10]:")
    print(str(arr))

    # 3. Advanced Array Methods
    arr.resize(12, 99)
    arr.setValue(-1, 1, 4)

    print("\nArray after resize(12, 99) and setValue(-1, 1, 4):")
    print(str(arr))

    # 4. NumPy Interoperability
    np_array: np.ndarray = np.from_dlpack(arr)
    np_array[0] = 5000

    print("\nValue in arr[0] after changing NumPy view:")
    print(arr[0])


if __name__ == "__main__":
    main()

import tempfile

from pytnl.containers import Array


def main() -> None:
    # 1. Create and size a 1D Array of floats
    a = Array[float]()
    a.setSize(10)

    # 2. Initialize the array elements
    size: int = a.getSize()
    for i in range(size):
        a[i] = float(i * 10)

    print(f"Array initialized with size: {size}")

    # 3. Iterate the array
    for elem in a:
        print(elem)

    # 4. Check if the array is empty
    print(f"Is the array empty? {a.empty()}")

    # 5. forAllElements / forElements — not yet available in Python bindings
    # a.forAllElements(f)
    # a.forElements(BEGIN, END, f)

    # 6. Access elements by index
    for i in range(a.getSize()):
        print(f"Element at position {i}: {a[i]}")

    # 7. Set individual elements
    for i in range(a.getSize()):
        a.setElement(i, 2 * i)
    print(f"After setElement: {a}")

    # 8. Set a range of elements to a value
    a.setValue(1, 0, 3)
    print(f"After setValue(1, 0, 3): {a}")

    # 9. Save and load from a binary file
    with tempfile.NamedTemporaryFile(suffix=".tnl") as tmp:
        a.save(tmp.name)
        b = Array[float]()
        b.load(tmp.name)
        print(f"Loaded array: {b}")

    # 10. Reset the array
    print(f"Before reset — empty? {a.empty()}")
    a.reset()
    print(f"After reset: {a}")
    print(f"Is the array empty? {a.empty()}")


if __name__ == "__main__":
    main()

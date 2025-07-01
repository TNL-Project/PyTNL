from pytnl.containers import NDArray

# Create a 2D array of floats
a = NDArray[2, float]()
a.setSizes(3, 4)

# Initialize the array elements
shape = a.getSizes()
for i in range(shape[0]):
    for j in range(shape[1]):
        a[i, j] = i + j


# Define a function for evaluation
def f(i: int, j: int) -> None:
    print(f"{[i, j]}:  {a[i, j] = }")


# Evaluate a function for all indices of the array
a.forAll(f)

# Print the memory layout of the array
print(list(a.getStorageArray()))

# Convert the array to a NumPy array
np = a.as_numpy()
print(np)
print(type(np), np.shape, np.dtype)

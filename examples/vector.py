from pytnl.containers import Vector

# Create two vectors of floats with the same size
size = 10
a = Vector[float](size)
b = Vector[float](size)

# Initialize the vector elements
for i in range(size):
    a[i] = 1.0
    b[i] = i + 42.0

# Compute a third vector
c = 2 * a + b

# Convert the vector to a Python list
print(list(c))

# Convert the vector to a NumPy array
np = c.as_numpy()
print(np)

# Modify the NumPy array
np[:] = 0.0

# Print the elements in the TNL vector
print(list(c))

# Slicing works on the TNL vector too
a[0 : size // 2] -= b[0 : size // 2] + c[0 : size // 2]
print(list(a))

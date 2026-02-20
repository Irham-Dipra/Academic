import numpy as np

# Create an example array 'x'
x = np.arange(6)
x = x.reshape((2, 3))
print("Original array x:")
print(x)
# Output:
# Original array x:
# [[0 1 2]
#  [3 4 5]]

# Create a new array of ones with the same shape and type as 'x'
y = np.ones_like(x)
print("\nNew array y (using ones_like(x)):")
print(y)
# Output:
# New array y (using ones_like(x)):
# [[1 1 1]
#  [1 1 1]]

# Example with a different data type
z = np.arange(3, dtype=float)
print("\nOriginal array z (float):")
print(z)
# Output:
# Original array z (float):
# [0. 1. 2.]

w = np.ones_like(z)
print("\nNew array w (using ones_like(z)):")
print(w)
# Output:
# New array w (using ones_like(z)):
# [1. 1. 1.]

import numpy as np
import matplotlib.pyplot as plt

# a = np.array([2, 3])

# a = np.add(a, 5)

# print(a)

# b = np.zeros((3, 4, 5), int)
# print(b)
# # print(b[0][1])
# print(b.shape)
# print(b.ndim)
# print(b.size)

# c = np.full((3, 3), 7)
# print(c)

# d = np.arange(1, 10, 2)
# print(d)

# e = np.linspace(0, 10, 5)
# print(e)
# e = np.rint(e).astype(int)
# print(e)

# f = np.random.randint(1, 10, 1)
# print(f)

# a = np.array([1, 2])
# b = np.array([1, 3])
# a = a ** b
# print(a)

plt.figure(figsize=(10, 4))
plt.stem([1, 2, 3], [4, 5, 6])
plt.grid(True)
plt.show()
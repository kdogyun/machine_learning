import numpy as np
import matplotlib.pyplot as plt


def activation_1(n):
    return 1 / (1 + np.exp(-n))

def activation_2(n):
    return n


p = np.arange(-2, 2, 0.01)

W1 = np.array([[10], [10]], float)
b1 = np.array([[-10], [10]], float)

W2 = np.array([1, 1], float)
b2 = 0

a1 = activation_1( W1 * p + b1)
a2 = activation_2( np.dot(W2, a1) + b2)

plt.plot(p, a2)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def activation(n):
    return ((n >= 0) + -1 * (n < 0 )).astype(np.int)

def aa(idx, n):
    return (C[idx,0] * n + C[idx,2]) * -1 * C[idx,1]

W1 = np.array([[1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1],
               [1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1]]
              , float).transpose()
b1 = np.array([-2, 3, 0.5, 0.5, -1.75, 2.25, -3.25, 3.75, 6.25, -5.75, -4.75]
              , float).transpose()

W2 = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
               [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1]]
              , float)
b2 = np.array([-3, -3, -3, -3]
              , float).transpose()

W3 = np.array([1, 1, 1, 1], float)
b3 = np.array([3], float)

x = float(input())
y = float(input())
custom = np.array([x,y], float).transpose()

a1 = activation(np.dot(W1, custom) + b1)
a2 = activation(np.dot(W2, a1) + b2)
a3 = np.dot(W3, a2) + b3
    
b1 = b1.reshape(-1,1)
C = np.hstack((W1, b1))

p = np.arange(0, 5, 0.01)

plt.figure(figsize=(10,5))
for i in range(C.shape[0]):
    plt.plot(p, aa(i, p))
if a3 == 1:
    plt.plot(x,y,'ko')
else:
    plt.plot(x,y,'yo')

plt.xlim(0,5)
plt.ylim(0,2.5)
plt.show()


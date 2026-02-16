import numpy as np
import matplotlib.pyplot as plt


def construct_A_centered_5pt(n, h):
    w = np.array([11/12,-14/3,19/2,-26/3,35/12], dtype=float)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for offset, weight in zip([0, 1, 2, 3, 4], w):
            j = i + offset
            if 0 <= j < n:
                A[i, j] += weight
    return A / h**2


def f(x):
    return np.exp(np.cos(x))*(np.sin(x)**2-np.cos(x))

# domain and BCs
a, b = 0.0, 2*np.pi
ua, ub = 0.0, 0.0
alpha, beta = 4, 0

N = 50                      # number of interior unknowns
h = (b - a) / (N + 1)
x_int = a + (np.arange(1, N+1) * h)

A = construct_A_centered_5pt(N, h)
rhs = f(x_int)
rhs[0]  -= alpha/h**2
rhs[-1] -= beta/h**2


u = np.linalg.solve(A, rhs)
print(u)


plt.plot(x_int, u, label="numerical u")
plt.legend()
plt.show()

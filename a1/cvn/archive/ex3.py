import numpy as np
import matplotlib.pyplot as plt

def construct_A_centered_5pt(n, h):
    w = np.array([-1/12, 4/3, -5/2, 4/3, -1/12], dtype=float)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for offset, weight in zip([-2, -1, 0, 1, 2], w):
            j = i + offset
            if 0 <= j < n:
                A[i, j] += weight
    return A / h**2

def f(x):
    return np.exp(np.cos(x))

# domain and BCs
a, b = 0.0, 2*np.pi
ua, ub = 0.0, 0.0

N = 50                      # number of interior unknowns
h = (b - a) / (N + 1)
x_int = a + (np.arange(1, N+1) * h)

A = construct_A_centered_5pt(N, h)
rhs = f(x_int)


A3 = (np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)) / h**2
rhs3 = f(x_int)
rhs3[0] -= ua / h**2
rhs3[-1] -= ub / h**2

u_int = np.linalg.solve(A3, rhs3)

# build full solution including boundaries
x = np.linspace(a, b, N+2)
u = np.zeros(N+2)
u[0], u[-1] = ua, ub
u[1:-1] = u_int

# verify: discrete second derivative ~ f(x) on interior
u_xx_num = (u[:-2] - 2*u[1:-1] + u[2:]) / h**2

plt.plot(x[1:-1], u_xx_num, label="numerical u''")
plt.plot(x[1:-1], f(x[1:-1]), '--', label="f(x)=exp(cos x)")
plt.legend()
plt.show()

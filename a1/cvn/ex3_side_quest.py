import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, eye, kron

# Exact solution
def u_exact(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

# f = Δu for the chosen exact solution
def f_rhs(x, y):
    term1 = -32 * np.pi**2 * np.sin(4 * np.pi * (x + y))
    term2 = -16 * np.pi**2 * (x**2 + y**2) * np.cos(4 * np.pi * x * y)
    return term1 + term2

def poisson5(m):
    e = np.ones(m)
    S = spdiags([e, -2*e, e], [-1, 0, 1], m, m, format='csr')
    I = eye(m, format='csr')
    A = kron(I, S) + kron(S, I)
    A = (m + 1)**2 * A   # since h = 1/(m+1), 1/h^2 = (m+1)^2
    return A.tocsr()

def construct_b(m, f, u_boundary):
    h = 1.0 / (m + 1)
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)

    b = np.zeros((m, m))

    for i in range(m):
        for j in range(m):
            xi = x[j]
            yj = y[i]
            b[i, j] = f(xi, yj)
            if i == 0:      # bottom boundary y = 0
                b[i, j] -= u_boundary(xi, 0.0) / h**2
            if i == m - 1:  # top boundary y = 1
                b[i, j] -= u_boundary(xi, 1.0) / h**2
            if j == 0:      # left boundary x = 0
                b[i, j] -= u_boundary(0.0, yj) / h**2
            if j == m - 1:  # right boundary x = 1
                b[i, j] -= u_boundary(1.0, yj) / h**2

    return b.ravel()

# Number of interior points
m = 10

A = poisson5(m)
b = construct_b(m, f_rhs, u_exact)

# Solve
u_vec = np.linalg.solve(A.toarray(), b)
u_num = u_vec.reshape((m, m))

# Interior grid for plotting / error check
h = 1.0 / (m + 1)
x = np.linspace(h, 1 - h, m)
y = np.linspace(h, 1 - h, m)
X, Y = np.meshgrid(x, y)

u_ex = u_exact(X, Y)
err = np.max(np.abs(u_num - u_ex))
print("Max error:", err)

# Plot numerical solution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, u_num, cmap='viridis')
ax.set_title("Numerical solution (5-point stencil)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# Optional: plot error
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, np.abs(u_num - u_ex), cmap='magma')
ax.set_title("Absolute error")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
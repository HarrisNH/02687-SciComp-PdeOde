import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import ex3_a as fcb
import ex3_b as fca

a = 0.5

psi = lambda x: 20 * np.pi * x**3
psidot = lambda x: 3 * 20 * np.pi * x**2
psiddot = lambda x: 2 * 3 * 20 * np.pi * x

f = lambda x: -20 + a * psiddot(x) * np.cos(psi(x)) - a * psidot(x)**2 * np.sin(psi(x))
u = lambda x: 1 + 12 * x - 10 * x**2 + a * np.sin(psi(x))

m = 255
h = 1 / (m + 1)

# Sparse tridiagonal finite difference matrix
A = diags(
    diagonals=[np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)],
    offsets=[-1, 0, 1],
    shape=(m, m),
    format="csr"
) / h**2

X = np.linspace(h, 1 - h, m)
F = f(X)

# Boundary condition adjustments
F[0] = F[0] - u(0) / h**2
F[-1] = F[-1] - u(1) / h**2

Uhat = u(X)
Ehat = spsolve(A, F) - Uhat

# Jacobi pieces
M = np.diag(A.diagonal())
N = M - A.toarray()
G = np.linalg.solve(M, N)
b = np.linalg.solve(M, F)

omega = 2 / 3

U2 = 1 + 2 * X

plt.ion()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i in range(10):
    U2 = (1 - omega) * U2 + omega * (G @ U2 + b)
    E2 = U2 - Uhat

    axes[0].cla()
    axes[0].plot(X, Uhat, 'b-', label='Exact')
    axes[0].plot(X, U2, 'gx', label='Iterate')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('U')
    axes[0].set_title(f'Iter={i+1:4d}')
    axes[0].tick_params(labelsize=12)

    axes[1].cla()
    axes[1].plot(X, Ehat, 'b-', label='Exact error')
    axes[1].plot(X, E2, 'gx', label='Current error')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('E')
    axes[1].set_title(f'Iter={i+1:4d}')
    axes[1].tick_params(labelsize=12)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.pause(1)

input("Press Enter to continue...")

# Calculate residual
r = F - A @ U2

# Coarsen
m_coarse = (m - 1) // 2
h_coarse = 1 / (m_coarse + 1)
r_coarse = r[1::2]   # MATLAB 2:2:end -> Python 1::2
assert len(r_coarse) == m_coarse

A_coarse = diags(
    diagonals=[np.ones(m_coarse - 1), -2 * np.ones(m_coarse), np.ones(m_coarse - 1)],
    offsets=[-1, 0, 1],
    shape=(m_coarse, m_coarse),
    format="csr"
) / h_coarse**2

# Solve the coarse problem directly
e_coarse = spsolve(A_coarse, -r_coarse)

# Project back to fine grid
e = np.zeros_like(r)
e[1::2] = e_coarse

for i in range(0, m, 2):   # MATLAB 1:2:m -> Python 0,2,4,...
    e_left = e[i - 1] if i > 0 else 0
    e_right = e[i + 1] if i < m - 1 else 0
    e[i] = (e_left + e_right) / 2

U2 = U2 - e
E2 = U2 - Uhat

axes[0].cla()
axes[0].plot(X, Uhat, 'b-', label='Exact')
axes[0].plot(X, U2, 'gx', label='Corrected')
axes[0].set_xlabel('x')
axes[0].set_ylabel('U')
axes[0].set_title('After coarse grid projection')
axes[0].tick_params(labelsize=12)

axes[1].cla()
axes[1].plot(X, Ehat, 'b-', label='Exact error')
axes[1].plot(X, E2, 'gx', label='Corrected error')
axes[1].set_xlabel('x')
axes[1].set_ylabel('E')
axes[1].set_title('After coarse grid projection')
axes[1].tick_params(labelsize=12)

fig.patch.set_facecolor('white')
plt.tight_layout()
plt.pause(0.1)

input("Press Enter to continue...")

# Smooth the error again
for i in range(10):
    U2 = (1 - omega) * U2 + omega * (G @ U2 + b)
    E2 = U2 - Uhat

    axes[0].cla()
    axes[0].plot(X, Uhat, 'b-', label='Exact')
    axes[0].plot(X, U2, 'gx', label='Iterate')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('U')
    axes[0].set_title(f'Iter={i+1:4d}')
    axes[0].tick_params(labelsize=12)

    axes[1].cla()
    axes[1].plot(X, Ehat, 'b-', label='Exact error')
    axes[1].plot(X, E2, 'gx', label='Current error')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('E')
    axes[1].set_title(f'Iter={i+1:4d}')
    axes[1].tick_params(labelsize=12)

    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.pause(1)

plt.ioff()
plt.show()
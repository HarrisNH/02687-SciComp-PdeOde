import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import cg, LinearOperator

# Exact solution
def u_exact(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

# derivative of u
def f_rhs(x, y):
    term1 = -32 * np.pi**2 * np.sin(4 * np.pi * (x + y))
    term2 = -16 * np.pi**2 * (x**2 + y**2) * np.cos(4 * np.pi * x * y)
    return term1 + term2

def Amult(U, m):

    h = (m + 1)**2
    U_reshape = U.reshape((m, m))

    AU = np.zeros_like(U_reshape)

    AU = 4 * U_reshape

    #neighbors 
    AU[:-1, :] -= U_reshape[1:, :] #below
    AU[1:, :] -= U_reshape[:-1, :] #above
    AU[:, :-1] -= U_reshape[:, 1:] #right
    AU[:, 1:] -= U_reshape[:, :-1] #left

    AU *= 

    # scaling by the factor 
    AU *= h

    return -AU.ravel()


def construct_b(m, f, u_boundary):
    h = 1.0 / (m + 1)
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)

    X, Y = np.meshgrid(x,y)
    b = f(X, Y)
    b[0, :] -= u_boundary(x, 0)/h**2
    b[-1, :] -= u_boundary(x, 1)/h**2
    
    b[:, 0] -= u_boundary(0, y)/h**2
    b[:, -1] -= u_boundary(1, y)/h**2

    return b.ravel()


m = 1000
F = construct_b(m, f_rhs, u_exact)

Aop = LinearOperator(
    shape=(m*m, m*m),
    matvec=lambda U: Amult(U, m),
    dtype=float
)

U_sol, exit_code = cg(Aop, -F, atol=1e-10)
U_sol = U_sol.reshape((m,m))

print("exit_code =", exit_code)
print("solution =", U_sol)

# Interior grid for plotting / error check
h = 1.0 / (m + 1)
x = np.linspace(h, 1 - h, m)
y = np.linspace(h, 1 - h, m)
X, Y = np.meshgrid(x, y)


# Plot numerical solution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, U_sol, cmap='viridis')
ax.set_title("Numerical solution (5-point stencil)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
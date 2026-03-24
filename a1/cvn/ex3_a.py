import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import cg, LinearOperator


# Exact solution
def u_exact(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)

def u_exact_grid():
    global m
    h = 1.0 / (m + 1)
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)
    X, Y = np.meshgrid(x, y)
    res = u_exact(X, Y).ravel()
    return -res

# derivative of u
def f_rhs(x, y):
    term1 = -32 * np.pi**2 * np.sin(4 * np.pi * (x + y))
    term2 = -16 * np.pi**2 * (x**2 + y**2) * np.cos(4 * np.pi * x * y)
    return term1 + term2


def Amult(U, m):

    h = 1/(m+1)

    U_reshape = U.reshape((m, m))

    AU = 4 * U_reshape.copy()

    # neighbors
    AU[:-1, :] -= U_reshape[1:, :]  # below
    AU[1:, :] -= U_reshape[:-1, :]  # above
    AU[:, :-1] -= U_reshape[:, 1:]  # right
    AU[:, 1:] -= U_reshape[:, :-1]  # left

    # scaling factor 
    AU *= 1/h**2

    return AU.ravel()


def construct_b(m, f, u_boundary):
    h = 1.0 / (m + 1)
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)

    X, Y = np.meshgrid(x, y)
    b = f(X, Y)
    b[0, :] -= u_boundary(x, 0) / h**2
    b[-1, :] -= u_boundary(x, 1) / h**2

    b[:, 0] -= u_boundary(0, y) / h**2
    b[:, -1] -= u_boundary(1, y) / h**2

    return b.ravel()


residuals = []
errors = []
def residual_change(xk):
    rk = -F - Aop.matvec(xk)
    residuals.append(np.max(rk))
    errors.append(np.max(u_exact_grid()-xk))
    return residuals

m = 100
F = construct_b(m, f_rhs, u_exact)

Aop = LinearOperator(shape=(m * m, m * m), matvec=lambda U: Amult(U, m), dtype=float)

if __name__ == "__main__":
    U_sol, exit_code = cg(-Aop, -F, tol=1e-14, atol=1e-14, callback=residual_change, maxiter=10_000)
    U_sol = U_sol.reshape((m,m))

    print("exit_code =", exit_code)

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
    plt.savefig("img/ex3_a_solution.png")

    # --- Residual log-log analysis ---
    residuals = np.array(residuals)
    iters = np.arange(1, len(residuals) + 1)

    plt.figure(figsize=(8, 6))
    plt.loglog(iters, residuals, "o-", label="CG residual")
    plt.loglog(iters, errors, "o-", label="CG errors")


    plt.xlabel("iteration")
    plt.ylabel("residual")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.savefig("img/ex3_a_error.png")

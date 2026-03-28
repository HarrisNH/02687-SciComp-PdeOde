import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
import ex3_a as fc_a
import ex3_b as fc_b


# exact solution and RHS
def u(x, y):
    return np.exp(np.pi * x) * np.sin(np.pi * y) + 0.5 * (x * y) ** 2


def f(x, y):
    return x**2 + y**2


def build_laplacian_2d(m):
    """
    Build the sparse matrix for the 2D discrete Laplacian.
    """
    h = 1.0 / (m + 1)

    T = diags(
        [np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)],
        [-1, 0, 1],
        shape=(m, m),
        format="csr"
    ) / h**2

    I = eye(m, format="csr")
    A = kron(I, T) + kron(T, I)
    return A

def coarsen(R, m):
    """
    Fine grid has m x m interior points, with m = 2^k - 1.
    Coarse grid has mc x mc interior points, mc = (m-1)//2.
    """
    if (m - 1) % 2 != 0:
        raise ValueError("Need m = 2^k - 1.")

    mc = (m - 1) // 2
    Rf = R.reshape((m, m))

    # coarse nodes coincide with odd/odd fine indices
    Rc = Rf[1::2, 1::2]

    assert Rc.shape == (mc, mc)
    return Rc.ravel()


def interpolate(Rc, m):

    mc = (m - 1) // 2
    Rc2 = Rc.reshape((mc, mc))

    #reformatting the number of available neighbors.
    C = np.zeros((mc + 2, mc + 2))
    C[1:-1, 1:-1] = Rc2

    R = np.zeros((m, m))

    #coarse nodes map to fine odd/odd point
    R[1::2, 1::2] = Rc2

    #points halffway in x, on coarse y-lines
    R[0::2, 1::2] = 0.5 * (C[:-1, 1:-1] + C[1:, 1:-1])

    # points halfway in y, on coarse x-lines
    R[1::2, 0::2] = 0.5 * (C[1:-1, :-1] + C[1:-1, 1:])

    #cell centers
    R[0::2, 0::2] = 0.25 * (
        C[:-1, :-1] + C[:-1, 1:] + C[1:, :-1] + C[1:, 1:]
    )

    return R.ravel()

def grid_points(k):
    return 2**k - 1

def main():
    m = 2**9 - 1
    U = np.zeros(m * m)
    F = fc_a.construct_b(m, fc_a.f_rhs, fc_a.u_exact)   # form the right-hand side
    epsilon = 1.0e-10
    omega = 2.0 / 3.0     

    for i in range(1, 101):
        R = F - fc_a.Amult(U, m)
        rel_resid = np.linalg.norm(R, 2) / np.linalg.norm(F, 2)
        print(f"*** Outer iteration: {i:3d}, rel. resid.: {rel_resid:e}")

        if rel_resid < epsilon:
            break

        U = Vcycle(U, omega, 3, m, F)
        plotU(m, U)
        plt.pause(1.5)

    plt.show()


def m_from_k(k):
    return 2**k - 1


def run_solver_for_k(k, epsilon=1e-10, maxiter=100):
    m = m_from_k(k)
    U = np.zeros(m * m)
    F = fc_a.construct_b(m, fc_a.f_rhs, fc_a.u_exact)
    omega = 2.0 / 3.0

    for i in range(1, maxiter + 1):
        R = F - fc_a.Amult(U, m)
        rel_resid = np.linalg.norm(R, 2) / np.linalg.norm(F, 2)

        if rel_resid < epsilon:
            return i

        U = Vcycle(U, omega, 3, m, F)

    return None


def iterations_test():
    k_values = [2, 3, 4, 5, 6, 7, 8, 9]
    m_values = [m_from_k(k) for k in k_values]
    iteration_counts = []

    for k in k_values:
        iters = run_solver_for_k(k)
        iteration_counts.append(iters)
        print(f"k={k}, m={m_from_k(k)}, iterations={iters}")

    plt.plot(m_values, iteration_counts, marker='o')
    plt.xlabel("Number of interior grid points m")
    plt.ylabel("Outer multigrid iterations")
    plt.title("Iterations needed to reach fixed tolerance")
    plt.grid(True)
    plt.show()



def Vcycle(U, omega, nsmooth, m, F):
    """
    Approximately solve: A * U = F
    """
    h = 1.0 / (m + 1)
    l2m = np.log2(m + 1)

    assert l2m == round(l2m)
    assert len(U) == m * m

    if m == 1:
        # if we are at the coarsest level
        # solve the only remaining equation directly!
        A1 = build_laplacian_2d(1)
        Unew = spsolve(A1, F).copy()

    else:

        # 1. pre-smooth the error - perform <nsmooth> Jacobi iterations
        for _ in range(nsmooth):
            U = fc_b.smooth(U, omega, m, F)

        # 2. calculate the residual
        r = F - fc_a.Amult(U, m)

        # 3. coarsen the residual
        rc = coarsen(r, m)

        # 4. recurse to Vcycle on a coarser grid
        mc = (m - 1) // 2
        Ecoarse = Vcycle(np.zeros(mc * mc), omega, nsmooth, mc, rc)

        # 5. interpolate the error
        e = interpolate(Ecoarse, m)

        # 6. update the solution given the interpolated error
        U = U + e

        # 7. post-smooth the error - perform <nsmooth> Jacobi iterations
        for _ in range(nsmooth):
            U = fc_b.smooth(U, omega, m, F)

        Unew = U.copy() 

    return Unew

def plotU(m, U):
    h = 1.0 / (m + 1)

    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)

    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = U.reshape((m, m)).T

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)

    ax.set_title("Computed solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("U")


if __name__ == "__main__":
    main()
    iterations_test()


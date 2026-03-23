import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags
import ex3_a as fcb
import ex3_b as fca


def coarsen(R, m):

    assert (m - 1) % 2 == 0, "Need m = 2^k - 1 so that mc = (m-1)//2 is integer."

    mc = (m - 1) // 2
    R2 = R.reshape((m, m))

    # Fine odd/odd points -> coarse grid
    Rc = R2[1::2, 1::2]

    assert Rc.shape == (mc, mc)
    return Rc.ravel()


def interpolate(Rc, m):

    assert (m - 1) % 2 == 0, "Need m = 2^k - 1 so that mc = (m-1)//2 is integer."

    mc = (m - 1) // 2
    Rc2 = Rc.reshape((mc, mc))

    R = np.zeros((m, m))
    
    R[1::2, 1::2] = Rc2

    # Fill odd/even indices by horizontal averaging with i odd, j even
    for i in range(1, m, 2):
        for j in range(0, m, 2):
            vals = []
            if j - 1 >= 0:
                vals.append(R[i, j - 1])
            if j + 1 < m:
                vals.append(R[i, j + 1])
            if vals:
                R[i, j] = sum(vals) / len(vals)

    # Fill even/odd indices by vertical averaging with i even, j odd
    for i in range(0, m, 2):
        for j in range(1, m, 2):
            vals = []
            if i - 1 >= 0:
                vals.append(R[i - 1, j])
            if i + 1 < m:
                vals.append(R[i + 1, j])
            if vals:
                R[i, j] = sum(vals) / len(vals)

    # Fill even/even indices by averaging the four diagonal neighbors with i even, j even
    for i in range(0, m, 2):
        for j in range(0, m, 2):
            vals = []
            if i - 1 >= 0 and j - 1 >= 0:
                vals.append(R[i - 1, j - 1])
            if i - 1 >= 0 and j + 1 < m:
                vals.append(R[i - 1, j + 1])
            if i + 1 < m and j - 1 >= 0:
                vals.append(R[i + 1, j - 1])
            if i + 1 < m and j + 1 < m:
                vals.append(R[i + 1, j + 1])
            if vals:
                R[i, j] = sum(vals) / len(vals)

    return R.ravel()


def build_A_1d(m):
    h = 1.0 / (m + 1)
    A = diags(
        diagonals=[np.ones(m - 1), -2 * np.ones(m), np.ones(m - 1)],
        offsets=[-1, 0, 1],
        shape=(m, m),
        format="csr"
    ) / h**2
    return A


def main():

    m = 63   # choose m = 2^k - 1
    omega = 2 / 3

    # Right-hand side and exact solution from ex3_a
    F = fcb.construct_b(m, fcb.f_rhs, fcb.u_exact)

    h = 1.0 / (m + 1)
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)
    X, Y = np.meshgrid(x, y)
    Uhat = fcb.u_exact(X, Y).ravel()

    # Initial guess
    U2 = np.zeros(m * m)

    # Pre-smoothing
    for _ in range(10):
        U2 = fca.smooth(U2, omega, m, F)
    
    # ex3_a.Amult(U,m) computes -A U so residual is F + Amult(U,m)
    r = F + fcb.Amult(U2, m)

    # Restrict residual to coarse grid
    rc = coarsen(r, m)
    mc = (m - 1) // 2

    # Coarse-grid solve A_c e_c = -r_c , note negative r_c because Amult uses -A
    # We build 2D Poisson matrix A_c explicitly here
    e1 = np.ones(mc)
    T = diags([e1, -2 * e1, e1], offsets=[-1, 0, 1], shape=(mc, mc), format="csr")
    I = diags([np.ones(mc)], [0], shape=(mc, mc), format="csr")
    h_c = 1.0 / (mc + 1)
    A_c = -(np.kron(np.eye(mc), T.toarray()) + np.kron(T.toarray(), np.eye(mc))) / h_c**2

    e_c = np.linalg.solve(A_c, -rc)

    # Interpolate correction back to fine grid
    e = interpolate(e_c, m)

    # Correct
    U2 = U2 - e

    # Post-smoothing
    for _ in range(10):
        U2 = fca.smooth(U2, omega, m, F)

    # Plot final solution and error
    U2_grid = U2.reshape((m, m))
    E2_grid = (U2 - Uhat).reshape((m, m))

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, U2_grid, cmap='viridis')
    ax1.set_title("Multigrid corrected solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, E2_grid, cmap='viridis')
    ax2.set_title("Error after coarse-grid correction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
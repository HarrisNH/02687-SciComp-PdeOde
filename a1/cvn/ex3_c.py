import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import spsolve
import ex3_a as fc_a
import ex3_b as fc_b


# ============================================================
# Fine-grid operator and smoother
# ============================================================


def build_laplacian_2d(m):
    """
    Build the sparse matrix for the 2D discrete Laplacian Δ_h.
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


# ============================================================
# Transfer operators
# ============================================================

def coarsen(R, m):
    """
    Restriction by injection.

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

    assert (m - 1) % 2 == 0, "Need m = 2^k - 1 so that mc = (m-1)//2 is integer."

    mc = (m - 1) // 2
    Rc2 = Rc.reshape((mc, mc))

    # Bilinear interpolation with zero Dirichlet boundary extension.
    # This keeps the 1/2 and 1/4 weights near the boundary instead of
    # renormalizing by the number of available neighbors.
    C = np.zeros((mc + 2, mc + 2))
    C[1:-1, 1:-1] = Rc2

    R = np.zeros((m, m))

    # coarse nodes map to fine odd/odd points
    R[1::2, 1::2] = Rc2

    # points halfway in x, on coarse y-lines
    R[0::2, 1::2] = 0.5 * (C[:-1, 1:-1] + C[1:, 1:-1])

    # points halfway in y, on coarse x-lines
    R[1::2, 0::2] = 0.5 * (C[1:-1, :-1] + C[1:-1, 1:])

    # cell centers
    R[0::2, 0::2] = 0.25 * (
        C[:-1, :-1] + C[:-1, 1:] + C[1:, :-1] + C[1:, 1:]
    )

    return R.ravel()


# ============================================================
# Two-grid cycle
# ============================================================

def two_grid_cycle(U, m, F, omega=2/3, nu1=3, nu2=3):
    """
    One 2-grid correction step:
      pre-smooth
      restrict residual
      solve coarse error equation
      prolong correction
      correct
      post-smooth
    """
    # pre-smoothing
    for _ in range(nu1):
        U = fc_b.smooth(U, omega, m, F)

    # fine-grid residual: r = F - Δ_h U
    r = F - fc_a.Amult(U, m)

    # restrict to coarse grid
    mc = (m - 1) // 2
    rc = coarsen(r, m)

    # coarse-grid solve: Δ_{2h} e_c = r_c
    A_c = build_laplacian_2d(mc)
    e_c = spsolve(A_c, rc)

    # prolongate and correct
    e = interpolate(e_c, m)
    U = U + e

    # post-smoothing
    for _ in range(nu2):
        U = fc_b.smooth(U, omega, m, F)

    return U


# ============================================================
# Driver
# ============================================================

def main():
    m = 63                  # must be 2^k - 1
    omega = 2 / 3
    n_cycles = 12
    nu1 = 3
    nu2 = 3

    h = 1.0 / (m + 1)
    x = np.linspace(h, 1 - h, m)
    y = np.linspace(h, 1 - h, m)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, fc_a.u_exact(X, Y), cmap='viridis')
    ax.set_title("Numerical solution (5-point stencil)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
    
    U_exact = fc_a.u_exact(X, Y).ravel()
    F = fc_a.construct_b(m, fc_a.f_rhs, fc_a.u_exact)

    # reference discrete solution
    A = build_laplacian_2d(m)
    U_ref = spsolve(A, F)

    # initial guess
    U = np.zeros(m * m)

    residual_history = []
    error_to_discrete_history = []

    for k in range(n_cycles):
        U = two_grid_cycle(U, m, F, omega=omega, nu1=nu1, nu2=nu2)

        r = F - fc_a.Amult(U, m)
        residual_norm = np.linalg.norm(r)
        discrete_error_norm = np.linalg.norm(U - U_ref)

        residual_history.append(residual_norm)
        error_to_discrete_history.append(discrete_error_norm)

        print(
            f"cycle {k+1:2d}: "
            f"||r||_2 = {residual_norm:.4e}, "
            f"||U-U_ref||_2 = {discrete_error_norm:.4e}"
        )

    # reshape for plots
    U_grid = U.reshape((m, m))
    U_ref_grid = U_ref.reshape((m, m))
    U_exact_grid = U_exact.reshape((m, m))
    E_grid = U_grid - U_exact_grid

    # --------------------------------------------------------
    # plots
    # --------------------------------------------------------
    fig = plt.figure(figsize=(15, 4.5))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(X, Y, U_grid, cmap="viridis")
    ax1.set_title("2-grid solution")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_surface(X, Y, E_grid, cmap="viridis")
    ax2.set_title("Error vs exact solution")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.semilogy(residual_history, "o-", label=r"$\|r_k\|_2$")
    ax3.semilogy(error_to_discrete_history, "s-", label=r"$\|U_k-U_{\mathrm{ref}}\|_2$")
    ax3.set_title("Convergence history")
    ax3.set_xlabel("2-grid cycle")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()

    # optional comparison print
    print("\nFinal norms:")
    print(f"||U - U_ref||_2   = {np.linalg.norm(U - U_ref):.6e}")
    print(f"||U - U_exact||_2 = {np.linalg.norm(U - U_exact):.6e}")
    print("Note: the exact-solution error does not go to zero because of discretization error.")


if __name__ == "__main__":
    main()

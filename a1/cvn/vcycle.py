import numpy as np
import matplotlib.pyplot as plt


# exact solution and RHS
def u(x, y):
    return np.exp(np.pi * x) * np.sin(np.pi * y) + 0.5 * (x * y) ** 2


def f(x, y):
    return x**2 + y**2


def main():
    m = 2**6 - 1
    U = np.zeros(m * m)
    F = form_rhs(m, f, u)   # TODO: Form the right-hand side
    epsilon = 1.0e-10
    omega = 2.0 / 3.0       # example value; set this as needed

    for i in range(1, 101):
        R = F - Amult(U, m)
        rel_resid = np.linalg.norm(R, 2) / np.linalg.norm(F, 2)
        print(f"*** Outer iteration: {i:3d}, rel. resid.: {rel_resid:e}")

        if rel_resid < epsilon:
            break

        U = Vcycle(U, omega, 3, m, F)
        plotU(m, U)
        plt.pause(0.5)

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
        # TODO: solve the only remaining equation directly!
        Unew = U.copy()
    else:
        # 1. TODO: pre-smooth the error
        #    perform <nsmooth> Jacobi iterations

        # 2. TODO: calculate the residual

        # 3. TODO: coarsen the residual

        # 4. recurse to Vcycle on a coarser grid
        mc = (m - 1) // 2
        Ecoarse = Vcycle(np.zeros(mc * mc), omega, nsmooth, mc, Rcoarse)

        # 5. TODO: interpolate the error

        # 6. TODO: update the solution given the interpolated error

        # 7. TODO: post-smooth the error
        #    perform <nsmooth> Jacobi iterations

        Unew = U.copy()  # placeholder until TODOs are implemented

    return Unew


def plotU(m, U):
    h = 1.0 / (m + 1)

    # MATLAB:
    # x=linspace(1/h,1-1/h,m);
    # y=linspace(1/h,1-1/h,m);
    #
    # That looks like a typo in the MATLAB code.
    # For an interior grid, this is almost certainly what was intended:
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


# -------------------------------------------------------------------
# Placeholder functions: these must be implemented by you
# -------------------------------------------------------------------

def form_rhs(m, f, u):
    raise NotImplementedError("form_rhs(m, f, u) has not been implemented yet.")


def Amult(U, m):
    raise NotImplementedError("Amult(U, m) has not been implemented yet.")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def T2(m):
    e = np.ones(m)
    return sp.diags([e[:-1], -2 * e, e[:-1]], [-1, 0, 1])


def stencil_9_A(m):
    e = np.ones(m)
    S = sp.spdiags([-e, -10 * e, -e], [-1, 0, 1], m, m, format="csr")
    I = sp.spdiags([-0.5 * e, e, -0.5 * e], [-1, 0, 1], m, m, format="csr")
    A = sp.kron(I, S) + sp.kron(S, I)
    return A


def stencil_5_A(m):
    T = T2(m)
    A = sp.kron(sp.identity(m), T) + sp.kron(T, sp.identity(m))

    return A


def f_poiss(x, y):
    pi = np.pi
    u_xx = -16 * pi**2 * np.sin(4 * pi * (x + y)) - 16 * pi**2 * (y**2) * np.cos(
        4 * pi * x * y
    )
    u_yy = -16 * pi**2 * np.sin(4 * pi * (x + y)) - 16 * pi**2 * (x**2) * np.cos(
        4 * pi * x * y
    )
    return u_xx + u_yy


def laplacian_of_f(x, y):
    pi = np.pi
    return (
        64
        * pi**2
        * (
            (4 * pi**2 * (x**2 + y**2) ** 2 - 1) * np.cos(4 * pi * x * y)
            + (8 * pi * x * y) * np.sin(4 * pi * x * y)
            + (16 * pi**2) * np.sin(4 * pi * (x + y))
        )
    )


def idx(i, j, m):
    return (j - 1) * m + (i - 1)


def stencil_9_b(m, x, y):
    b = np.zeros(m**2)
    h = 1 / (m + 1)
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            k = idx(i, j, m)
            b[k] = (6 * h**2) * (
                f_poiss(x[i], y[j]) + 1 / 12 * h**2 * laplacian_of_f(x[i], y[j])
            )

            # x y neighbors w. weight 4
            for ii, jj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if ii == 0 or ii == m + 1 or jj == 0 or jj == m + 1:
                    b[k] -= 4 * u_exact(x[ii], y[jj])

            # diag neighbors w. weight 1
            for ii, jj in [
                (i - 1, j - 1),
                (i + 1, j + 1),
                (i + 1, j - 1),
                (i - 1, j + 1),
            ]:
                if ii == 0 or ii == m + 1 or jj == 0 or jj == m + 1:
                    b[k] -= u_exact(x[ii], y[jj])
    return b


def stencil_5_b(m, x, y):
    # Find b first
    b = np.zeros(m**2)
    h = 1 / (m + 1)
    for i in range(1, m + 1):
        for j in range(1, m + 1):
            k = idx(i, j, m)
            b[k] = h**2 * f_poiss(x[i], y[j])

            if i == 1:
                b[k] -= u_exact(x[0], y[j])
            if j == 1:
                b[k] -= u_exact(x[i], y[0])
            if i == m:
                b[k] -= u_exact(x[m + 1], y[j])
            if j == m:
                b[k] -= u_exact(x[i], y[m + 1])
    return b


def u_exact(x, y):
    return np.sin(4 * np.pi * (x + y)) + np.cos(4 * np.pi * x * y)


def convergence_plotter(
    m0,
    levels,
    t_start,
    t_end,
    approx_u_func,
    exact_u_func=None,
    plt_show=False,
    plt_title="default",
    exact=2,
):

    m0 = m0  # start interior count
    levels = levels
    m_arr = (m0 + 1) * (2 ** np.arange(levels)) - 1  # halve h each step
    h_arr = 1 / (m_arr + 1)
    d_arr = np.zeros(len(h_arr) - 1)
    if exact_u_func == None:
        exact_flag = 0
    else:
        exact_flag = 1

    for i, m in enumerate(m_arr):
        if i != 0:
            u_prev = u_i

        x = np.linspace(t_start, t_end, m + 2)
        y = np.linspace(t_start, t_end, m + 2)
        X_int, Y_int = np.meshgrid(x[1:-1], y[1:-1], indexing="xy")
        X_int_1d, Y_int_1d = X_int.ravel(order="C"), Y_int.ravel(order="C")
        try:
            u_i = approx_u_func(m, x, y)
        except TypeError as e:
            print(e)
            break

        if i != 0:
            if exact_flag == 0:
                d_arr[i - 1] = np.linalg.norm(u_i[::2] - u_prev, ord=np.inf)
            else:
                d_arr[i - 1] = np.linalg.norm(
                    u_i - exact_u_func(X_int_1d, Y_int_1d), ord=np.inf
                )

    h_coarse = h_arr[:-1]  # matches d_arr
    mask = d_arr > 0  # avoid log(0) if it happens

    # fitted slope - try not to take the 3irst 3 values
    p, logC = np.polyfit(np.log(h_coarse[mask]), np.log(d_arr[mask]), 1)
    print(f"slope: {p:.3f}  (expect ~2)")

    # reference O(h^2) line anchored at the last point
    h0 = h_coarse[mask][-1]
    d0 = d_arr[mask][-1]
    ref = d0 * (h_coarse / h0) ** exact
    label_str = (
        "||u_h - u_{2h}|| (on coarse grid)"
        if exact_flag == 0
        else "||u_h - u_{exact}||"
    )
    plt.figure()
    plt.loglog(h_coarse, d_arr, "o-", label=label_str)
    plt.loglog(h_coarse, ref, "--", label=rf"reference $\propto h^{exact}$")

    plt.gca().invert_xaxis()  # optional: finer h to the right feels nicer to many
    plt.xlabel("h (coarse)")
    plt.ylabel("difference norm")
    plt.title(f"Convergence plot (slope ~ {p:.3f})")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(f"a1/harh/{plt_title}.png")
    plt.show(block=plt_show)
    return h_arr, d_arr


def find_u_5(m, x, y):
    A = stencil_5_A(m)
    b = stencil_5_b(m, x, y)
    u = spla.spsolve(A, b)
    return u


def find_u_9(m, x, y):
    A = stencil_9_A(m)
    b = stencil_9_b(m, x, y)
    u = spla.spsolve(A, b)
    return u


m0 = 3
levels = 7
t_start, t_end = 0, 1
convergence_plotter(
    m0, levels, t_start, t_end, find_u_5, u_exact, plt_show=False, plt_title="5_stencil"
)
convergence_plotter(
    m0,
    levels,
    t_start,
    t_end,
    find_u_9,
    u_exact,
    plt_show=True,
    plt_title="9_stencil",
    exact=4,
)

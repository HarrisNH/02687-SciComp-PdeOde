import numpy as np
import matplotlib.pyplot as plt


def jacobian_nonlinear(eps, h, um, u0, up):
    return np.array(
        [
            eps / h**2 - u0 / (2 * h),
            -2 * eps / h**2 + (up - um) / (2 * h) - 1,
            eps / h**2 + u0 / (2 * h),
        ]
    )


def g_func(eps, h, xm, x0, xp):
    return np.array(
        [
            (eps / h**2 - x0 / (2 * h)) * xm,
            (2 * eps / h**2 + (xp - xm) / (2 * h) - 1) * x0,
            (eps / h**2 - x0 / (2 * h)) * xp,
        ]
    )


def jacobian_linear(eps, h):
    # For operator: eps*u'' - u
    return np.array([eps / h**2, -(2 * eps / h**2 + 1.0), eps / h**2])


def func(x, eps):
    return -(eps + 1.0) * np.sin(x)


def non_lin_func(x, eps):
    return -(eps + 1) * np.sin(x) + np.sin(x) * np.cos(x)


def u_exact_sin(x):
    return np.sin(x)


def create_system(eps, grid):
    h = grid[1] - grid[0]
    N = grid.size
    A = np.zeros((N, N))
    b = func(grid, eps)

    aL, aC, aR = jacobian_linear(eps, h)
    for i in range(1, N - 1):
        A[i, i - 1] = aL
        A[i, i] = aC
        A[i, i + 1] = aR

    u0 = np.sin(grid[0])
    uN = np.sin(grid[-1])

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = u0

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = uN

    return A, b


def solve_lin_system(x):
    eps = 0.1
    A, b = create_system(eps, x)
    return np.linalg.solve(A, b)


# setup
m = 100
t0, T = 0.0, 1.0
eps = 0.1


grid = np.linspace(t0, T, m + 2)  # endpoints included
A, b = create_system(eps, grid)
u = np.linalg.solve(A, b)

# plt.plot(grid, u, label="numerical")
# plt.plot(grid, np.sin(grid), "--", label="exact sin(x)")
# plt.legend()
# plt.show()

##### ABOVE THIS LINE SOLVES THE LINEAR PART using sin(x) as the manufactured solution


def forcing_full(x, eps):
    return -(eps + 1.0) * np.sin(x) + np.sin(x) * np.cos(x)


def assemble_F_J(u, x, eps, s_func=None, bc_left=None, bc_right=None):
    h = x[1] - x[0]
    N = x.size

    if s_func is None:
        s = np.zeros_like(x)
    else:
        s = s_func(x, eps)
    if bc_left is None:
        bc_left = u[0]
    if bc_right is None:
        bc_right = u[-1]

    F = np.zeros(N)
    J = np.zeros((N, N))

    # Dirichlet BC equations
    F[0] = u[0] - bc_left
    J[0, 0] = 1.0

    F[-1] = u[-1] - bc_right
    J[-1, -1] = 1.0

    # interior equations after utilizing symmetric property
    for i in range(1, N - 1):
        um, u0i, up = u[i - 1], u[i], u[i + 1]

        ux = (up - um) / (2 * h)
        uxx = (up - 2 * u0i + um) / (h * h)

        # residual to manufactured solution for eps*u'' + u*(u'-1) = s
        F[i] = eps * uxx + u0i * (ux - 1.0) - s[i] 

        # jacobian function - differentiation of the stencil
        J[i, i - 1] = eps / (h * h) - u0i / (2 * h)
        J[i, i] = -2 * eps / (h * h) + (up - um) / (2 * h) - 1.0
        J[i, i + 1] = eps / (h * h) + u0i / (2 * h)

    return F, J


def newton_solve(
    x,
    eps=0.1,
    max_iter=30,
    tol=1e-10,
    s_func=None,
    bc_left=None,
    bc_right=None,
    u_init=None,
):
    # initial gues using manufactured solution
    if u_init is None:
        u = np.zeros_like(x)
    else:
        u = u_init(x).copy()
    if bc_left is not None:
        u[0] = bc_left
    if bc_right is not None:
        u[-1] = bc_right

    for k in range(max_iter):
        F, J = assemble_F_J(u, x, eps, s_func, bc_left, bc_right)

        # solving the Newton optimization J * delta = -F
        delta = np.linalg.solve(J, -F)
        u = u + delta

        if np.linalg.norm(delta, ord=np.inf) < tol:
            break

    return u


def solve_actual(x):
    return newton_solve(
        x,
        eps=0.1,
        s_func=None,
        bc_left=-1.0,
        bc_right=1.5,
    )


def solve_nonlin_mms_sin(x):
    eps = 0.1
    bound_left = np.sin(x[0])
    bound_right = np.sin(x[-1])
    u_init = u_exact_sin
    u = newton_solve(
        x,
        eps,
        s_func=forcing_full,
        bc_left=bound_left,
        bc_right=bound_right,
        u_init=u_init,
    )
    return u


# setup
m = 1000
t0, T = 0.0, 1.0
eps = 0.1
x = np.linspace(t0, T, m + 2)

bc_left = -1.0
bc_right = 1.5
u_num = newton_solve(x, eps, bc_left=bc_left, bc_right=bc_right)
# u_ex  = np.sin(x)

plt.plot(x, u_num, label="Newton (MMS)")
# plt.plot(x, u_ex, "--", label="exact sin(x)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(r"Solution function u(x) to $\epsilon$u''+u(u'-1)=0")
plt.grid()
plt.legend()
plt.show()


def convergence_plotter(
    m0,
    levels,
    t_start,
    t_end,
    approx_u_func,
    exact_u_func=None,
    plt_show=False,
    plt_title="default",
    expect=2,
):

    m0 = m0  # start interior count
    levels = levels
    m_arr = (m0 + 1) * (2 ** np.arange(levels)) - 1  # halve h each step
    h_arr = 1 / (m_arr + 1)
    d_arr = np.zeros(len(h_arr) - 1)

    if exact_u_func == None:
        exact_flag = 1
    else:
        exact_flag = 0
    for i, m in enumerate(m_arr):
        if i != 0:
            u_prev = u_i

        x = np.linspace(t_start, t_end, m + 2)
        try:
            u_i = approx_u_func(x)
        except TypeError as e:
            print(e)
            break

        if i != 0:
            if exact_flag == 1:
                d_arr[i - 1] = np.linalg.norm(u_i[::2] - u_prev, ord=np.inf)
            else:
                d_arr[i - 1] = np.linalg.norm(u_i - exact_u_func(x), ord=np.inf)

    h_coarse = h_arr[:-1]  # matches d_arr
    mask = d_arr > 0  

    # fitted slope 
    p, logC = np.polyfit(np.log(h_coarse[mask]), np.log(d_arr[mask]), 1)
    print(f"slope: {p:.3f}  (expect ~{expect})")

    # reference O(h^2) line anchored at the last point
    h0 = h_coarse[mask][-1]
    d0 = d_arr[mask][-1]
    ref = d0 * (h_coarse / h0) ** expect
    label_str = (
        "||u_h - u_{2h}|| (on coarse grid)"
        if exact_flag == 0
        else "||u_h - u_{exact}||"
    )
    plt.figure()
    plt.loglog(h_coarse, d_arr, "o-", label=label_str)
    plt.loglog(h_coarse, ref, "--", label=rf"reference $\propto h^{expect}$")
    plt.gca().invert_xaxis() 
    plt.xlabel("h (coarse)")
    plt.ylabel("difference norm")
    plt.title(f"Convergence plot (slope ~ {p:.3f})")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(f"a1/cvn/{plt_title}.png")
    plt.show(block=plt_show)
    return h_arr, d_arr


m = 10
levels = 9
t_start, t_end = 0.0, 1.0

print("Plot for MMS with sin(x) and only linear part:")
convergence_plotter(
    m,
    levels,
    t_start,
    t_end,
    approx_u_func=solve_lin_system,
    plt_show=True,
    plt_title="ex2b_MMS_sin_lin",
)
print("Solving MMS with Nonlinear parts using sin(x)")
convergence_plotter(
    m,
    levels,
    t_start,
    t_end,
    solve_nonlin_mms_sin,
    u_exact_sin,
    True,
    "ex2b_MMS_sin_full",
)
print("Solve actual system from exercise:")
convergence_plotter(
    m,
    levels,
    t_start,
    t_end,
    solve_actual,
    plt_show=True,
    plt_title="ex2b_f0",
    expect=4,
)

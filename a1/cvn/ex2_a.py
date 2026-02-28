import numpy as np
import matplotlib.pyplot as plt

def jacobian_nonlinear(eps, h, um, u0, up):
    return np.array([
        eps / h**2 - u0 / (2*h),
        -2*eps / h**2 + (up - um) / (2*h) - 1,
        eps / h**2 + u0 / (2*h),
    ])

def g_func(eps, h, xm, x0, xp):
    return np.array([
        (eps / h**2 - x0 / (2*h))*xm,
        (2*eps / h**2 + (xp - xm) / (2*h) - 1)*x0,
        (eps / h**2 - x0 / (2*h))*xp,
    ])


def jacobian_linear(eps, h):
    # For operator: eps*u'' - u
    return np.array([eps/h**2, -(2*eps/h**2 + 1.0), eps/h**2])

def func(x, eps):
    return -(eps + 1.0) * np.sin(x)

def non_lin_func(x, eps):
    return -(eps+1)*np.sin(x) + np.sin(x)*np.cos(x)

def create_system(eps, grid):
    h = grid[1] - grid[0]
    N = grid.size
    A = np.zeros((N, N))
    b = func(grid, eps)

    aL, aC, aR = jacobian_linear(eps, h)
    for i in range(1, N-1):
        A[i, i-1] = aL
        A[i, i]   = aC
        A[i, i+1] = aR

    u0 = np.sin(grid[0])
    uN = np.sin(grid[-1])

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = u0

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = uN

    return A, b
# setup
m = 100
t0, T = 0.0, 1.0
eps = 0.1

grid = np.linspace(t0, T, m+2)  # endpoints included
A, b = create_system(eps, grid)
u = np.linalg.solve(A, b)

# plt.plot(grid, u, label="numerical")
# plt.plot(grid, np.sin(grid), "--", label="exact sin(x)")
# plt.legend()
#plt.show()

##### ABOVE THIS LINE SOLVES THE LINEAR PART using sin(x) as the manufactured solution

def forcing_full(x, eps):
    return -(eps + 1.0)*np.sin(x) + np.sin(x)*np.cos(x)

def assemble_F_J(u, x, eps):
    h = x[1] - x[0]
    N = x.size

    s = forcing_full(x, eps)*0

    F = np.zeros(N)
    J = np.zeros((N, N))

    u0 = -1#np.sin(x[0])
    uN = 1.5#np.sin(x[-1])

    # Dirichlet BC equations
    F[0] = u[0] - u0
    J[0, 0] = 1.0

    F[-1] = u[-1] - uN
    J[-1, -1] = 1.0

    #interior equations after utilizing symmetric property
    for i in range(1, N-1):
        um, u0i, up = u[i-1], u[i], u[i+1]

        ux  = (up - um)/(2*h)
        uxx = (up - 2*u0i + um)/(h*h)

        # residual to manufactured solution for eps*u'' + u*(u'-1) = s
        F[i] = eps*uxx + u0i*(ux - 1.0) - s[i]

        #jacobian function - differentiation of the discretized version w.r.t. U_{i-1}, U_{i}, U_{i+1} respectively
        J[i, i-1] = eps/(h*h) - u0i/(2*h)
        J[i, i]   = -2*eps/(h*h) + (up - um)/(2*h) - 1.0
        J[i, i+1] = eps/(h*h) + u0i/(2*h)

    return F, J

def newton_solve(x, eps, max_iter=30, tol=1e-10):
    # initial guess: manufactured solution is fine, or zeros
    u = np.zeros_like(x)
    # enforce BC in initial guess - note the sin(x[0]) is only used for the mnaufactured solution, we set to correct BC
    #  when things looks correct
    u[0] = -1#np.sin(x[0])
    u[-1] = 1.5#np.sin(x[-1])

    for k in range(max_iter):
        F, J = assemble_F_J(u, x, eps)

        #solving the Newton optimization J * delta = -F
        delta = np.linalg.solve(J, -F)
        u = u + delta

        if np.linalg.norm(delta, ord=np.inf) < tol:
            break

    return u

# setup
m = 1000
t0, T = 0.0, 1.0
eps = 0.1
x = np.linspace(t0, T, m+2)

u_num = newton_solve(x, eps)
#u_ex  = np.sin(x)

plt.plot(x, u_num, label="Newton (MMS)")
#plt.plot(x, u_ex, "--", label="exact sin(x)")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title(r"Solution function u(x) to $\epsilon$u''+u(u'-1)=0")
plt.grid()
plt.legend()
plt.show()

#print("max error:", np.max(np.abs(u_num - u_ex)))


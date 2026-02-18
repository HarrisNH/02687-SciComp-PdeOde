import numpy as np
import math
import matplotlib.pyplot as plt


def fdcoeffV(k, xbar, x):

    x = np.asarray(x, dtype=float)
    n = x.size

    A = np.ones((n, n), dtype=float)
    xrow = x - xbar

    for i in range(1, n):
        A[i, :] = (xrow**i) / math.factorial(i)

    b = np.zeros(n, dtype=float)
    b[k] = 1.0

    c = np.linalg.solve(A, b)
    return c


def fdcoeffF(k: int, xbar, x):
    # k'th derivative of u is being approximated
    # derivative evaluated at xbar
    # x are grid points
    # length(x) > k
    n = len(x)
    x = np.asarray(x, dtype=float)
    if k >= n:
        raise Exception("len(x) must be greater than k")

    m = k

    c1 = 1.0  # becomes product of distances between stencil pts from prev. steps
    c4 = x[0] - xbar  # offset from xbar
    C = np.zeros((n, m + 1), dtype=float)  # row for each point, col for each derivative
    C[0, 0] = 1.0  # if only one point then u(xbar) = u(xbar)

    for i in range(1, n):  # add stencil point one at a time
        mn = min(i, m)  # only i stencil points available, so max order of deriv
        c2 = 1.0  # used to calc distances from new point to all older points
        c5 = c4  # old point's offset
        c4 = x[i] - xbar  # new point's offset

        for j in range(0, i):  # loop over old points
            c3 = x[i] - x[j]  # dist new point to each old point
            c2 = c2 * c3  # all distances multiplied
            if j == i - 1:
                for s in range(mn, 0, -1):  # create stencil weights for x(i1)
                    # stencil weights for s1 w. derivs less than max order of deriv
                    C[i, s] = c1 * (s * C[i - 1, s - 1] - c5 * C[i - 1, s]) / c2

                C[i, 0] = -c1 * c5 * C[i - 1, 0] / c2

            for s in range(mn, 0, -1):
                C[j, s] = (c4 * C[j, s] - s * C[j, s - 1]) / c3
            C[j, 0] = (c4 * C[j, 0]) / c3
        c1 = c2
    c = C[:, k]
    return c


def u(x):
    return np.exp(np.cos(x))


def u_dd_exact_at_0():
    # u''(x) = (-cos x + sin^2 x) * exp(cos x), so u''(0) = -e
    return -math.e


# ---- Exercise (c): approximate u''(0) on an equidistant grid ----
xbar = 0.0
k = 2

exact = u_dd_exact_at_0()
print("Exact u''(0) =", exact)

num = 10
err = np.zeros(num - 1)
err_centered = np.zeros(num - 1)
hvals = np.zeros(num - 1)

i = 0
for m in range(2, num):
    h = 1 / m**2
    hvals[i] = h

    x_nodes = xbar + h * np.arange(0, 5)
    x_nodes2 = xbar + h * np.arange(-2, 3)

    c = fdcoeffF(k, xbar, x_nodes)
    c_centered = fdcoeffF(k, xbar, x_nodes2)

    approx = np.dot(c, u(x_nodes))
    approx_centered = np.dot(c_centered, u(x_nodes2))

    err[i] = abs(approx - exact)
    err_centered[i] = abs(approx_centered - exact)

    i += 1

print(f"h={h: .3e}   approx={approx: .12f}   err={err[-1]: .3e}")

mask = np.isfinite(err) & (err > 0) & np.isfinite(hvals) & (hvals > 0)

h_plot = hvals[mask]
err_plot = err[mask]
errc_plot = err_centered[mask]

# anchor constants so the reference lines pass through the last data point (smallest h)
C3 = err_plot[-1] / (h_plot[-1]**3)
C4 = errc_plot[-1] / (h_plot[-1]**4)

ref_h3 = C3 * h_plot**3
ref_h4 = C4 * h_plot**4


p = np.polyfit(np.log(hvals[mask]), np.log(err[mask]), 1)[0]
p_cent = np.polyfit(np.log(hvals[mask]), np.log(err_centered[mask]), 1)[0]
print("Estimated convergence rate p_non_centered =", p)
print("Estimated convergence rate p_centered =", p_cent)

<<<<<<< HEAD
plt.loglog(hvals, err, "o-", label="not centered")
plt.loglog(hvals, err_centered, "o-", label="centered")
=======
plt.figure()
plt.loglog(h_plot, err_plot, 'o-', label='not centered')
plt.loglog(h_plot, errc_plot, 'o-', label='centered')
plt.loglog(h_plot, ref_h3, '--', label=r'ref: $C h^3$')
plt.loglog(h_plot, ref_h4, '--', label=r'ref: $C h^4$')
>>>>>>> 3fd7a9b (fix)
plt.xlabel("h")
plt.ylabel("error")
plt.legend()
plt.title(r"Convergence of $u''(0)$, $u(x)=e^{\cos x}$")
plt.savefig("img/ex3_error.png", dpi=200, bbox_inches="tight")


plt.plot(approx)
plt.savefig("img/ex3_approx_func.png")


## To check why we see order 4 conv. for uncentered
a_4 = 11 / 12
a_3 = -14 / 3
a_2 = 19 / 2
a_1 = -26 / 3
a0 = 35 / 12

f1 = -4 * a_4 - 3 * a_3 - 2 * a_2 - a_1
f2 = 16 * a_4 + 9 * a_3 + 4 * a_2 + a_1
f3 = -64 * a_4 - 27 * a_3 - 8 * a_2 - a_1
f4 = 256 * a_4 + 81 * a_3 + 16 * a_2 + a_1
f5 = -1024 * a_4 - 243 * a_3 - 32 * a_2 - a_1
print(f"f1: {f1} f2: {f2} f3: {f3} f4: {f4} f5: {f5}")

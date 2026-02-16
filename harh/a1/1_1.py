import numpy as np


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


a_vec = fdcoeffF(2, 0, [-4, -3, -2, -1, 0])
print(a_vec)

import sympy as sp

a_4, a_3, a_2, a_1, a0, h = sp.symbols("a_{-4} a_{-3} a_{-2} a_{-1} a_0 h ")
h = 1
eq1 = sp.Eq(0, a_4 + a_3 + a_2 + a_1 + a0)
eq2 = sp.Eq(0, -4 * a_4 - 3 * a_3 - 2 * a_2 - 1 * a_1)
eq3 = sp.Eq(h ** (-2), 1 / 2 * (16 * a_4 + 9 * a_3 + 4 * a_2 + 1 * a_1))
eq4 = sp.Eq(0, 1 / 6 * (-64 * a_4 - 27 * a_3 - 8 * a_2 - 1 * a_1))
eq5 = sp.Eq(0, 1 / 24 * (256 * a_4 + 81 * a_3 + 16 * a_2 + 1 * a_1))


solutions = sp.solve([eq1, eq2, eq3, eq4, eq5], [a_4, a_3, a_2, a_1, a0])
print(solutions)

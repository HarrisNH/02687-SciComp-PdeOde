import fdcoeffF as fd
import numpy as np

xbar = 0
n = 2
h = 1 / 4


def equi_cent_grid_1d(xbar, n, h):
    """
    n: number of points to one side
    """
    x_array = np.linspace(xbar - n * h, xbar + n * h, 2 * n + 1)
    return np.array(x_array)


x_vec = equi_cent_grid_1d(xbar, n, h)
print(x_vec.shape)

a_vec = fd.fdcoeffF(2, xbar, x_vec)


def f(x):
    return np.exp(np.cos(x))


print(x_vec)
print(a_vec, x_vec, f(x_vec))
x0_approx = np.dot(a_vec, f(x_vec))
x0_exact = -np.exp(1)
print(x0_approx, x0_exact)

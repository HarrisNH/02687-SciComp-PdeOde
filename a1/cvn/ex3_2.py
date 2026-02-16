import numpy as np
import math
import matplotlib.pyplot as plt

def fdcoeffV(k, xbar, x):

    x = np.asarray(x, dtype=float)
    n = x.size

    A = np.ones((n, n), dtype=float)
    xrow = (x - xbar) 

    for i in range(1, n):  
        A[i, :] = (xrow ** i) / math.factorial(i)

    b = np.zeros(n, dtype=float)
    b[k] = 1.0  

    c = np.linalg.solve(A, b)
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

num = 30
err = np.zeros(num-1)
hvals = np.zeros(num-1)

i = 0
for m in [ i for i in range(2, num)]:
    h = 1 / m**2
    print(m**2)
    hvals[i] = h

    x_nodes = xbar + h * np.arange(0, 5)

    c = fdcoeffV(k, xbar, x_nodes)
    approx = np.dot(c, u(x_nodes))
    err[i] = abs(approx - exact)

    i += 1

print(f"h={h: .3e}   approx={approx: .12f}   err={err[-1]: .3e}")



mask = np.isfinite(err) & (err > 0) & np.isfinite(hvals) & (hvals > 0)

p = np.polyfit(np.log(hvals[mask]), np.log(err[mask]), 1)[0]
print("Estimated convergence rate p =", p)

plt.loglog(hvals, err, 'o-')
plt.xlabel("h")
plt.ylabel("error")
plt.title("Convergence of errors with f(x) = exp(cos(x))")
plt.savefig("ex3_error.png")

plt.plot(approx)
plt.savefig("ex3_approx_func.png")
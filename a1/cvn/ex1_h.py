import numpy as np
import matplotlib.pyplot as plt

def u(x):
    return np.exp(np.cos(x))

def u_xx(x):
    return np.exp(np.cos(x)) * (np.sin(x)**2 - np.cos(x))

xbar = 0.3  # choose non-special point
exact = u_xx(xbar)

s_vals = np.arange(2, 6)
hvals = 2.0 ** (-s_vals)

err_base = []
err_re = []

for h in hvals:
    # nodes
    xm2, xm1, x0, xp1, xp2 = xbar + h*np.array([-2, -1, 0, 1, 2])
    # base 5-point stencil (O(h^4))
    D_h = (-1/12*u(xm2) + 4/3*u(xm1) - 5/2*u(x0) + 4/3*u(xp1) - 1/12*u(xp2)) / h**2

    # same with h/2
    hh = h/2
    xm2, xm1, x0, xp1, xp2 = xbar + hh*np.array([-2, -1, 0, 1, 2])
    D_h2 = (-1/12*u(xm2) + 4/3*u(xm1) - 5/2*u(x0) + 4/3*u(xp1) - 1/12*u(xp2)) / hh**2

    # Richardson with p=4 -> O(h^6)
    D_re = (16*D_h2 - D_h) / 15

    err_base.append(abs(D_h - exact))
    err_re.append(abs(D_re - exact))

err_base = np.array(err_base)
err_re = np.array(err_re)

# fit slopes in truncation regime (before roundoff upturn)
jmin = np.argmin(err_re)
use = np.arange(0, jmin+1)

p_base = np.polyfit(np.log(hvals[use]), np.log(err_base[use]), 1)[0]
p_re   = np.polyfit(np.log(hvals[use]), np.log(err_re[use]), 1)[0]
print("slope base ~", p_base, "(expect ~4)")
print("slope RE   ~", p_re,   "(expect ~6)")
plt.loglog(hvals, err_base, "o-", label=r"5-pt ($\alpha$,$\beta$)=(2,2)")
plt.loglog(hvals, err_re, "o-", label=r"RE ($\alpha$,$\beta$)=(2,2)")
plt.loglog(hvals, err_base[0]*(hvals/hvals[0])**4, "--",label=r"ref $h^4$")
plt.loglog(hvals, err_re[0]*(hvals/hvals[0])**6, "--", label=r"ref $h^6$")

plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("error")
plt.legend()
plt.savefig("img/ex1_h_error.png", dpi=200)







xbar = 0.3  # choose non-special point
exact = u_xx(xbar)

s_vals = np.arange(2, 6)
hvals = 2.0 ** (-s_vals)

err_base = []
err_re = []

for h in hvals:
    # nodes
    xm4, xm3, xm2, xm1, x0 = xbar + h*np.array([-4,-3,-2,-1,0])
    # base 5-point stencil (O(h^4))
    D_h = (11/12*u(xm4) - 14/3*u(xm3) + 19/2*u(xm2) - 26/3*u(xm1) + 35/12*u(x0)) / h**2

    # same with h/2
    hh = h/2
    xm4, xm3, xm2, xm1, x0 = xbar + hh*np.array([-4,-3,-2,-1,0])
    D_h2 = (11/12*u(xm4) - 14/3*u(xm3) + 19/2*u(xm2) - 26/3*u(xm1) + 35/12*u(x0)) / hh**2

    # Richardson with p=3 -> O(h^4)
    D_re = (8*D_h2 - D_h) / 7

    err_base.append(abs(D_h - exact))
    err_re.append(abs(D_re - exact))

err_base = np.array(err_base)
err_re = np.array(err_re)

# fit slopes in truncation regime (before roundoff upturn)
jmin = np.argmin(err_re)
use = np.arange(0, jmin+1)

p_base = np.polyfit(np.log(hvals[use]), np.log(err_base[use]), 1)[0]
p_re   = np.polyfit(np.log(hvals[use]), np.log(err_re[use]), 1)[0]
print("slope base ~", p_base, "(expect h^3)")
print("slope RE   ~", p_re,   "(expect h^4)")

plt.loglog(hvals, err_base, "o-", label=r"5-pt ($\alpha$,$\beta$)=(4,0)")
plt.loglog(hvals, err_re,   "o-", label=r"RE ($\alpha$,$\beta$)=(4,0)")
plt.loglog(hvals, err_base[-1]*(hvals/hvals[-1])**3, "--", label="ref $h^3$")
plt.loglog(hvals, err_re[-1]*(hvals/hvals[-1])**4, "--", label="ref $h^4$")

plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("error")
plt.legend()

plt.savefig("img/ex1_h_error_uncentered.png", dpi=200)








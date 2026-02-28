import numpy as np
import matplotlib.pyplot as plt
from ex3_2 import fdcoeffF

def u(x):
    return np.exp(np.cos(x))

def u_x(x):
    return -np.sin(x) * np.exp(np.cos(x))

if __name__ == "__main__":
    # point of the stencil center around zero
    xbar = 0.001
    k = 1
    # Evaluating true functional value
    exact = u_x(xbar)
    # creating a sequence for convergence with h_i = 1/2^h
    s_vals = np.arange(2, 6)  
    hvals = 1.0 / (2.0 ** s_vals)

    err_central = np.zeros_like(hvals)
    err_rich = np.zeros_like(hvals)

    for i, h in enumerate(hvals):
        #creating a set of x values which get tighter around zero with h getting smaller
        x_nodes = xbar + h * np.array([-1.0, 0.0, 1.0])
        c = fdcoeffF(k, xbar, x_nodes)
        D0_h = np.dot(c, u(x_nodes))

        # Same central difference but with step h/2
        hh = h / 2.0
        x_nodes_h2 = xbar + hh * np.array([-1.0, 0.0, 1.0])
        c_h2 = fdcoeffF(k, xbar, x_nodes_h2)
        D0_h2 = np.dot(c_h2, u(x_nodes_h2))

        # Richardson extrapolation with p=2
        D_RE = (4.0 * D0_h2 - D0_h) / 3.0

        err_central[i] = abs(D0_h - exact)
        err_rich[i] = abs(D_RE - exact)


    mask = (hvals > 0) & np.isfinite(err_central) & (err_central > 0) & np.isfinite(err_rich) & (err_rich > 0)

    jmin = np.argmin(err_rich)
    use = np.arange(0, jmin+1)

    p_r = np.polyfit(np.log(hvals[use]), np.log(err_rich[use]), 1)[0]
    p_c = np.polyfit(np.log(hvals[use]), np.log(err_central[use]), 1)[0]

    print("Estimated convergence rate p_central    =", p_c)
    print("Estimated convergence rate p_richardson =", p_r)


    h_plot = hvals[mask]
    ec = err_central[mask]
    er = err_rich[mask]

    j = np.argmin(er)    
    C4 = er[j] / (h_plot[j]**4)
    C2 = ec[j] / (h_plot[j]**2)


    plt.figure()
    plt.loglog(h_plot, ec, "o-", label="central (expected O(h^2))")
    plt.loglog(h_plot, er, "o-", label="Richardson (expected O(h^4))")
    plt.loglog(h_plot, C2 * h_plot**2, "--", label=r"ref: $C h^2$")
    plt.loglog(h_plot, C4 * h_plot**4, "--", label=r"ref: $C h^4$")
    plt.xlabel("h")
    plt.ylabel("error")
    plt.legend()
    plt.title(r"Convergence of $u'(0)$, $u(x)=e^{\cos x}$")
    plt.tight_layout()
    plt.savefig("img/ex3_g_error.png", dpi=200)

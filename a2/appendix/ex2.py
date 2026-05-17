
import numpy as np
import matplotlib.pyplot as plt

# ---------- Problem data ----------
EPS = 0.1
ALPHAS = np.array([1.0, 4.0, 16.0])
A_COEFS = np.array([1.0, 1.0, 1.0])
B_COEFS = np.array([0.0, 0.0, 0.0])


def u_exact(x, t):
    """Exact solution evaluated at array x, scalar t."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    for alpha, a, b in zip(ALPHAS, A_COEFS, B_COEFS):
        out += np.exp(-EPS * alpha**2 * t) * (a * np.cos(alpha * x) + b * np.sin(alpha * x))
    return out


def eta(x):
    return u_exact(x, 0.0)


def gL(t):
    return float(u_exact(np.array([-1.0]), t))


def gR(t):
    return float(u_exact(np.array([1.0]), t))


def ftcs_solve(N, T, r_target=0.4):

    h = 2.0 / N
    x = -1.0 + h * np.arange(N + 1)

    k_target = r_target * h**2 / EPS
    M = int(np.ceil(T / k_target))
    k = T / M
    r = EPS * k / h**2  

    u = eta(x)
    for n in range(M):
        t_next = (n + 1) * k
        u_new = np.empty_like(u)
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        u_new[0] = gL(t_next)
        u_new[-1] = gR(t_next)
        u = u_new

    return x, u, k, r, M


def main():
    T_final = 0.2

    N_list = [40, 80, 160, 320, 640]

    hs, errors, ks, rs = [], [], [], []

    print(f"{'N':>5} {'h':>10} {'k':>12} {'r':>8} {'M':>8} {'L_inf error':>14}")
    print("-" * 65)
    for N in N_list:
        x, u, k, r, M = ftcs_solve(N, T_final, r_target=0.4)
        u_true = u_exact(x, T_final)
        err = float(np.max(np.abs(u - u_true)))
        hs.append(2.0 / N)
        errors.append(err)
        ks.append(k)
        rs.append(r)
        print(f"{N:>5} {2.0/N:>10.5f} {k:>12.6e} {r:>8.4f} {M:>8d} {err:>14.6e}")

    hs = np.array(hs)
    errors = np.array(errors)

    print("\nObserved rate (log2 of error ratio per halving):")
    for i in range(1, len(errors)):
        rate = np.log2(errors[i - 1] / errors[i])
        print(f"  N = {N_list[i-1]:>4} -> {N_list[i]:>4}:   rate = {rate:.3f}")

    slope, intercept = np.polyfit(np.log(hs), np.log(errors), 1)
    print(f"\nLog-log linear fit slope: {slope:.3f}")

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(hs, errors, "o-", label="FTCS error")
    ref = errors[0] * (hs / hs[0]) ** 2
    ax.loglog(hs, ref, "k--", label=r"$\mathcal{O}(h^2)$ reference")
    ax.set_xlabel("h")
    ax.set_ylabel(r"$\max_i\,|u(x_i, T) - u_i^M|$")
    ax.set_title(f"FTCS convergence on $[-1,1]$ at $T={T_final}$, $r\\approx{rs[0]:.2f}$")
    ax.legend()
    ax.grid(True, which="both", ls=":", alpha=0.6)
    fig.tight_layout()

    out_png = "ftcs_convergence.png"
    #fig.savefig(out_png, dpi=130)
    fig.show()
    print(f"\nSaved convergence plot to {out_png}")


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt

EPSILON = 0.1  

def u_exact(x, t, eps=EPSILON):
    return -np.tanh((x + 0.5 - t) / (2.0 * eps)) + 1.0

def eta(x):
    return u_exact(x, 0.0)

def gL(t):
    return u_exact(-1.0, t)

def gR(t):
    return u_exact( 1.0, t)

def burgers_solve(N, T, eps=EPSILON, cfl_adv=0.8, cfl_diff=0.4):
    
    h = 2.0 / N
    x = -1.0 + h * np.arange(N + 1)
    u = eta(x)
    u[0]  = gL(0.0)
    u[-1] = gR(0.0)

    t = 0.0
    while t < T:
        u_max = max(np.max(np.abs(u)), 1e-12)
        dt_adv  = cfl_adv  * h / u_max
        dt_diff = cfl_diff * h**2 / eps
        dt = min(dt_adv, dt_diff, T - t)

        u_new = u.copy()

        for i in range(1, N):
            ui = u[i]
            ul = u[i - 1]
            ur = u[i + 1]

            adv = ui * (ui - ul) / h   # backward difference

            # central diffusion
            diff = eps * (ur - 2.0 * ui + ul) / h**2

            u_new[i] = ui + dt * (-adv + diff)

        t_next    = t + dt
        u_new[0]  = gL(t_next)
        u_new[-1] = gR(t_next)

        u = u_new
        t = t_next

    return x, u

def convergence_study(T_final=0.3):
    N_list = [40, 80, 160, 320, 640]

    print("=" * 65)
    print(f"Viscous Burgers convergence  (eps={EPSILON}, T={T_final})")
    print(f"{'N':>5} {'h':>10} {'L_inf error':>14}")
    print("-" * 35)

    hs, errors = [], []
    for N in N_list:
        x, u = burgers_solve(N, T_final)
        err  = float(np.max(np.abs(u - u_exact(x, T_final))))
        hs.append(2.0 / N)
        errors.append(err)
        print(f"{N:>5} {2.0/N:>10.5f} {err:>14.6e}")

    hs     = np.array(hs)
    errors = np.array(errors)

    print("\nObserved convergence rates:")
    for i in range(1, len(errors)):
        rate = np.log2(errors[i-1] / errors[i])
        print(f"  N={N_list[i-1]:>4} -> N={N_list[i]:>4}:  rate = {rate:.3f}")

    return hs, errors

def main():
    T_final = 0.3

    hs, errors = convergence_study(T_final)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── left: solution comparison ─────────────────────────────────────────
    ax1 = axes[0]
    for N, ls, col in [(40, "--", "C1"), (160, "-.", "C2"), (640, ":", "C3")]:
        x, u = burgers_solve(N, T_final)
        ax1.plot(x, u, ls=ls, color=col, lw=1.4, label=f"Numerical  N={N}")

    x_fine = np.linspace(-1, 1, 2000)
    ax1.plot(x_fine, u_exact(x_fine, T_final), "k-", lw=2, label="Exact")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$u$")
    ax1.set_title(f"Burgers solution at $T={T_final}$,  $\\epsilon={EPSILON}$")
    ax1.legend(fontsize=8); ax1.grid(True, ls=":", alpha=0.6)

    # ── middle: convergence plot ──────────────────────────────────────────
    ax2 = axes[1]
    ax2.loglog(hs, errors, "o-b", lw=1.5, label="$L_\\infty$ error")
    ref1 = errors[0] * (hs / hs[0])**1
    ref2 = errors[0] * (hs / hs[0])**2
    ax2.loglog(hs, ref1, "k--",  label=r"$\mathcal{O}(h)$")
    ax2.loglog(hs, ref2, "k:",   label=r"$\mathcal{O}(h^2)$")
    ax2.set_xlabel("$h$"); ax2.set_ylabel(r"$L_\infty$ error")
    ax2.set_title("Convergence")
    ax2.legend(); ax2.grid(True, which="both", ls=":", alpha=0.6)

    # ── right: evolution in time ──────────────────────────────────────────
    ax3 = axes[2]
    N_plot = 320
    x_p = -1.0 + (2.0 / N_plot) * np.arange(N_plot + 1)
    for t_snap, col in [(0.0, "C0"), (0.1, "C1"), (0.2, "C2"), (0.3, "C3")]:
        _, u_snap = burgers_solve(N_plot, t_snap)
        ax3.plot(x_p, u_snap, color=col, lw=1.3, label=f"$t={t_snap}$")

    ax3.set_xlabel("$x$"); ax3.set_ylabel("$u$")
    ax3.set_title(f"Solution evolution  $\\epsilon={EPSILON}$")
    ax3.legend(); ax3.grid(True, ls=":", alpha=0.6)

    fig.suptitle("Viscous Burgers",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("burgers.png", dpi=130)
    print("\nPlot saved to burgers.png")
    plt.show()


if __name__ == "__main__":
    main()
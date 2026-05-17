import numpy as np
import matplotlib.pyplot as plt

EPS    = 0.01 / np.pi
T_STAR = 1.6037 / np.pi
TARGET = -152.00516


def burgers_solve(N, T, eps=EPS, cfl_adv=0.3, cfl_diff=0.2):

    h = 2.0 / N
    x = -1.0 + h * np.arange(N + 1)

    u      = -np.sin(np.pi * x)
    u[0]   = 0.0
    u[-1]  = 0.0

    t = 0.0
    while t < T:
        u_max   = max(np.max(np.abs(u)), 1e-12)
        dt_adv  = cfl_adv  * h / u_max
        dt_diff = cfl_diff * h**2 / eps
        dt      = min(dt_adv, dt_diff, T - t)

        u_new = u.copy()

        for i in range(1, N):
            ui = u[i]
            ul = u[i - 1]
            ur = u[i + 1]

            # upwind advection — direction depends on sign of u
            if ui >= 0:
                adv = ui * (ui - ul) / h    # FTBS
            else:
                adv = ui * (ur - ui) / h    # FTFS

            diff     = eps * (ur - 2.0 * ui + ul) / h**2
            u_new[i] = ui + dt * (-adv + diff)

        u_new[0]  = 0.0
        u_new[-1] = 0.0

        u  = u_new
        t += dt

    return x, u

def derivative_at_zero(x, u):

    h  = x[1] - x[0]
    i0 = int(np.argmin(np.abs(x)))
    if x[i0] > 0 and i0 > 0:
        i0 -= 1
    return x[i0], (u[i0 + 1] - u[i0 - 1]) / (2.0 * h)

def main():
    N_list = [50, 100, 200, 400, 800, 1600, 3200]

    print(f"eps    = 0.01/pi = {EPS:.6f}")
    print(f"T      = 1.6037/pi = {T_STAR:.6f}")
    print(f"Target = {TARGET}\n")
    print(f"{'N':>6} {'h':>12} {'du/dx|_0':>16} {'error':>14}")
    print("-" * 55)

    hs, derivs = [], []
    for N in N_list:
        x, u    = burgers_solve(N, T_STAR)
        x0, dux = derivative_at_zero(x, u)
        err     = abs(dux - TARGET)
        hs.append(2.0 / N)
        derivs.append(dux)
        print(f"{N:>6} {2.0/N:>12.6f} {dux:>16.5f} {err:>14.6e}")

    hs     = np.array(hs)
    derivs = np.array(derivs)
    errors = np.abs(derivs - TARGET)

    print("\nConvergence rates:")
    for i in range(1, len(errors)):
        if errors[i] > 0 and errors[i-1] > 0:
            rate = np.log2(errors[i-1] / errors[i])
            print(f"  N={N_list[i-1]:>5} -> N={N_list[i]:>5}:  rate = {rate:.3f}")

    # ── plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: full solution for several N
    ax1 = axes[0]
    for N, ls, col in [(100, "--", "C1"), (400, "-.", "C2"), (1600, ":", "C3")]:
        xp, up = burgers_solve(N, T_STAR)
        ax1.plot(xp, up, ls=ls, color=col, lw=1.3, label=f"$N={N}$")
    ax1.axvline(0, color="k", ls=":", lw=0.8, label="$x=0$")
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$u$")
    ax1.set_title(f"Solution at $T = 1.6037/\\pi$")
    ax1.legend(fontsize=8); ax1.grid(True, ls=":", alpha=0.6)

    # right: convergence of the derivative estimate toward target
    ax2 = axes[1]
    ax2.loglog(hs, errors, "o-b", lw=1.5,
               label=r"$|\partial_x u|_{x=0} - \mathrm{target}|$")
    ref1 = errors[0] * (hs / hs[0])**1
    ref2 = errors[0] * (hs / hs[0])**2
    ax2.loglog(hs, ref1, "k--", label=r"$\mathcal{O}(h)$")
    ax2.loglog(hs, ref2, "k:",  label=r"$\mathcal{O}(h^2)$")
    ax2.set_xlabel("$h$")
    ax2.set_ylabel(r"error in $\partial_x u|_{x=0}$")
    ax2.set_title("Convergence of derivative estimate")
    ax2.legend(); ax2.grid(True, which="both", ls=":", alpha=0.6)

    fig.suptitle(
        r"Burgers — homogeneous BCs, $\eta(x)=-\sin(\pi x)$, "
        r"$\epsilon=0.01/\pi$",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("burgers_part3.png", dpi=130)
    print("\nPlot saved to burgers_part3.png")
    plt.show()


if __name__ == "__main__":
    main()
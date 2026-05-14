import numpy as np
import matplotlib.pyplot as plt


A = 0.5          # advection speed
CR_TARGET = 0.8  # Courant number used for FTBS (stable)


def u_exact(x, t):
    return np.sin(2.0 * np.pi * (x - A * t))


def ftbs_solve(N, T, cr=CR_TARGET):
    h  = 2.0 / N
    dt = cr * h / A
    M  = int(np.ceil(T / dt))
    dt = T / M
    cr_actual = A * dt / h

    x = -1.0 + h * np.arange(N)
    u = u_exact(x, 0.0)

    for _ in range(M):
        u_left = np.roll(u, 1) #smart function to roll the window of values ex [0,1,2,3] -> np.roll(x,2) -> [2,3,0,1]
        u = u - cr_actual * (u - u_left)

    return x, u, h, dt, cr_actual, M


def ftcs_solve(N, T, cr=CR_TARGET):
    h  = 2.0 / N
    dt = cr * h / A
    M  = int(np.ceil(T / dt))
    dt = T / M
    cr_actual = A * dt / h

    x = -1.0 + h * np.arange(N)
    u = u_exact(x, 0.0)

    for n in range(M):
        u_right = np.roll(u, -1)
        u_left  = np.roll(u,  1)
        u_new   = u - (cr_actual / 2.0) * (u_right - u_left)

        u = u_new

    return x, u, h, dt, cr_actual, M


def convergence_study(T_final=0.2):
    N_list = [40, 80, 160, 320, 640, 1280, 2560]

    print("=" * 65)
    print("FTBS convergence")
    print(f"{'N':>5} {'h':>10} {'c_r':>8} {'M':>8} {'L_inf error':>14}")
    print("-" * 55)

    hs_b, errs_b = [], []
    for N in N_list:
        x, u, h, dt, cr, M = ftbs_solve(N, T_final)
        err = float(np.max(np.abs(u - u_exact(x, T_final))))
        hs_b.append(h); errs_b.append(err)
        print(f"{N:>5} {h:>10.5f} {cr:>8.4f} {M:>8d} {err:>14.6e}")

    print("\nRates:")
    for i in range(1, len(errs_b)):
        print(f"  N={N_list[i-1]:>4}→{N_list[i]:>4}: "
              f"{np.log2(errs_b[i-1]/errs_b[i]):.3f}")

    print("\n" + "=" * 65)
    print("FTCS convergence  (expect blow-up / garbage errors)")
    print(f"{'N':>5} {'h':>10} {'c_r':>8} {'M':>8} {'L_inf error':>14}")
    print("-" * 55)

    hs_c, errs_c = [], []
    for N in N_list:
        x, u, h, dt, cr, M = ftcs_solve(N, T_final)
        err = float(np.max(np.abs(u - u_exact(x, T_final))))
        hs_c.append(h); errs_c.append(err)
        print(f"{N:>5} {h:>10.5f} {cr:>8.4f} {M:>8d} {err:>14.6e}")

    return (np.array(hs_b), np.array(errs_b),
            np.array(hs_c), np.array(errs_c))

def von_neumann_analysis(cr):

    xi = np.linspace(0, np.pi, 500)

    g_re = 1.0 - cr + cr * np.cos(xi)
    g_im = -cr * np.sin(xi)

    g_mod   = np.sqrt(g_re**2 + g_im**2)   # amplitude factor |g|
    phi_num = np.arctan2(g_im, g_re)    # numerical phase shift / step
    phi_ex  = -cr * xi # exact phase shift / step

    return xi, g_mod, phi_num, phi_ex


# ── 40-period prediction and simulation ──────────────────────────────────────
def period_analysis(cr=CR_TARGET, pts_per_wavelength=100, n_periods=40):

    lam  = 1.0     # wavelength of sin(2*pi*x)
    dx   = lam / pts_per_wavelength   # spatial step
    N    = pts_per_wavelength    # number of grid points (one wavelength)
    dt   = cr * dx / A      # time step from CFL
    T    = n_periods * lam / A   # total time
    M    = int(round(T / dt)) # exact number of steps
    dt   = T / M       # adjust to hit T exactly
    cr_actual = A * dt / dx

    # ── Von Neumann prediction for xi = 2*pi*dx (the resolved wavenumber) ──
    xi_wave = 2.0 * np.pi * dx   

    g_re    = 1.0 - cr_actual + cr_actual * np.cos(xi_wave)
    g_im    = -cr_actual * np.sin(xi_wave)
    g_mod   = np.sqrt(g_re**2 + g_im**2)
    phi_num = np.arctan2(g_im, g_re)
    phi_ex  = -cr_actual * xi_wave

    amp_predicted   = g_mod ** M
    phase_error_rad = M * (phi_num - phi_ex) 
    phase_error_dx  = phase_error_rad / (2.0 * np.pi) * lam  # in spatial units

    print("\n" + "=" * 65)
    print(f"Von Neumann prediction — {n_periods} wave periods, c_r = {cr_actual:.4f}")
    print(f"  dx = {dx:.4f},  dt = {dt:.6f},  M = {M} steps")
    print(f"  Resolved wavenumber xi = 2*pi*dx = {xi_wave:.6f} rad")
    print(f"  |g| per step          = {g_mod:.10f}")
    print(f"  Amplitude after {M} steps = |g|^M = {amp_predicted:.6f}")
    print(f"  Phase error per step  = {phi_num - phi_ex:.6e} rad")
    print(f"  Cumulative phase err  = {phase_error_rad:.6f} rad  "
          f"= {np.degrees(phase_error_rad):.4f} deg")
    print(f"  Spatial phase lag     ≈ {phase_error_dx:.6f}  (in x-units)")

    x = dx * np.arange(N)          
    u = np.sin(2.0 * np.pi * x)     

    for _ in range(M):
        u_left = np.roll(u, 1)
        u = u - cr_actual * (u - u_left)

    # exact solution at T
    u_ex = np.sin(2.0 * np.pi * (x - A * T))

    # predicted solution from Von Neumann
    u_pred = amp_predicted * np.sin(2.0 * np.pi * x + M * phi_num)

    return x, u, u_ex, u_pred, amp_predicted, phase_error_rad, M



def main():
    T_final = 0.2
    hs_b, errs_b, hs_c, errs_c = convergence_study(T_final)


    fig1, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.loglog(hs_b, errs_b, "o-b",  label="FTBS error")
    ax.loglog(hs_c, errs_c, "s--r", label="FTCS error (unstable)")
    ref = errs_b[0] * (hs_b / hs_b[0])
    ax.loglog(hs_b, ref, "k:", label=r"$\mathcal{O}(h)$ reference")
    ax.set_xlabel("$h$"); ax.set_ylabel(r"$L_\infty$ error")
    ax.set_title("Convergence comparison")
    ax.legend(fontsize=8); ax.grid(True, which="both", ls=":", alpha=0.6)

    ax2 = axes[1]
    N_plot = 160
    x_b, u_b, *_ = ftbs_solve(N_plot, T_final)
    ax2.plot(x_b, u_exact(x_b, T_final), "k-",  lw=1.5, label="Exact")
    ax2.plot(x_b, u_b,                   "b--", lw=1.2, label=f"FTBS  N={N_plot}")
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$u$")
    ax2.set_title(f"FTBS solution at $T={T_final}$")
    ax2.legend(); ax2.grid(True, ls=":", alpha=0.6); ax2.set_ylim(-1.5, 1.5)

    ax3 = axes[2]
    x_c, u_c, *_ = ftcs_solve(N_plot, T_final)
    u_c_clipped = np.clip(u_c, -5, 5)
    ax3.plot(x_c, u_exact(x_c, T_final), "k-",  lw=1.5, label="Exact")
    ax3.plot(x_c, u_c_clipped,           "r--", lw=1.2,
             label=f"FTCS  N={N_plot} (clipped)")
    ax3.set_xlabel("$x$"); ax3.set_ylabel("$u$")
    ax3.set_title(f"FTCS solution at $T={T_final}$ — unstable!")
    ax3.legend(); ax3.grid(True, ls=":", alpha=0.6); ax3.set_ylim(-1.5, 1.5)

    fig1.tight_layout()
    fig1.savefig("ftbs_ftcs_compare.png", dpi=130)

    # ── Figure 2: Von Neumann — |g| and phase speed vs wavenumber ────────────
    xi, g_mod, phi_num, phi_ex = von_neumann_analysis(CR_TARGET)

    fig2, (ax_a, ax_p) = plt.subplots(1, 2, figsize=(12, 5))

    # mark the resolved wavenumber for the 100-pts-per-wavelength case
    xi_mark = 2.0 * np.pi / 100.0

    ax_a.plot(xi / np.pi, g_mod, "b-", lw=2, label=r"FTBS $|g(\xi)|$")
    ax_a.axhline(1.0, color="k", ls="--", lw=1, label="Exact $|g|=1$")
    ax_a.axvline(xi_mark / np.pi, color="r", ls=":", lw=1.5,
                 label=r"$\xi = 2\pi/100$  (100 pts/$\lambda$)")
    ax_a.set_xlabel(r"$\xi / \pi$")
    ax_a.set_ylabel(r"$|g|$")
    ax_a.set_title(r"Amplitude factor $|g|$ — diffusion error")
    ax_a.legend(); ax_a.grid(True, ls=":", alpha=0.6)
    ax_a.set_ylim(0, 1.05)

    # numerical vs exact phase speed ratio  c_num/c_exact = phi_num / phi_ex
    with np.errstate(divide="ignore", invalid="ignore"):
        phase_ratio = np.where(np.abs(phi_ex) > 1e-12,
                               phi_num / phi_ex, 1.0)

    ax_p.plot(xi / np.pi, phase_ratio, "b-", lw=2, label=r"$c_\mathrm{num}/c_\mathrm{exact}$")
    ax_p.axhline(1.0, color="k", ls="--", lw=1, label="Exact ratio = 1")
    ax_p.axvline(xi_mark / np.pi, color="r", ls=":", lw=1.5,
                 label=r"$\xi = 2\pi/100$")
    ax_p.set_xlabel(r"$\xi / \pi$")
    ax_p.set_ylabel(r"$c_\mathrm{num} / c_\mathrm{exact}$")
    ax_p.set_title("Phase speed ratio — dispersion error")
    ax_p.legend(); ax_p.grid(True, ls=":", alpha=0.6)

    fig2.suptitle(f"Von Neumann analysis of FTBS,  $c_r = {CR_TARGET}$",
                  fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("ftbs_von_neumann.png", dpi=130)

    # ── Figure 3: 40-period simulation vs prediction ──────────────────────────
    (x_40, u_40, u_ex_40, u_pred_40,
     amp_pred, phase_err_rad, M_40) = period_analysis()

    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

    # left: overlay numerical, exact, and predicted
    ax4 = axes3[0]
    ax4.plot(x_40, u_ex_40,   "k-",  lw=2,   label="Exact")
    ax4.plot(x_40, u_pred_40, "g--", lw=1.8, label=f"Von Neumann prediction  (amp={amp_pred:.4f})")
    ax4.plot(x_40, u_40,      "b:",  lw=1.5, label="FTBS numerical")
    ax4.set_xlabel("$x$"); ax4.set_ylabel("$u$")
    ax4.set_title(f"After 40 wave periods  ($M={M_40}$ steps,  $c_r={CR_TARGET}$)")
    ax4.legend(fontsize=9); ax4.grid(True, ls=":", alpha=0.6)
    ax4.set_ylim(-1.3, 1.3)

    # right: error between numerical and exact
    ax5 = axes3[1]
    ax5.plot(x_40, u_40 - u_ex_40, "b-", lw=1.5, label="Numerical $-$ Exact")
    ax5.axhline(0, color="k", ls="--", lw=0.8)
    ax5.set_xlabel("$x$"); ax5.set_ylabel("error")
    ax5.set_title("Pointwise error after 40 periods")
    ax5.legend(); ax5.grid(True, ls=":", alpha=0.6)

    # annotate predicted amplitude loss and phase lag
    txt = (f"Predicted amplitude: {amp_pred:.4f}\n"
           f"Cumulative phase error: {np.degrees(phase_err_rad):.2f}°")
    ax4.text(0.02, 0.05, txt, transform=ax4.transAxes,
             fontsize=9, va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray"))

    fig3.suptitle("FTBS — 40-period diffusion & dispersion analysis",
                  fontsize=13, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("ftbs_40periods.png", dpi=130)

    print("\nAll plots saved.")
    plt.show()


if __name__ == "__main__":
    main()
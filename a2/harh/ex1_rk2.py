import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rk2(f, t0, tf, y0, h0=None):
    y = np.atleast_1d(y0).astype(float)
    t = t0

    if h0 is None:
        h = (tf - t0) / 100
    else:
        h = h0

    t_history = [t]
    y_history = [y.copy()]
    
    n_evals = 0
    while t < tf:
        n_evals += 2
        if t + h > tf:
            h = tf - t

        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        
        y = y + h * (1/2 * k1 + 1/2 * k2)
        t = t + h
    
        t_history.append(t)
        y_history.append(y.copy())

    return np.array(t_history), np.array(y_history), n_evals

def f(t, y):
    return y**2 - y**3

def libsol(t0, tf, delta, t_eval):
    sol = solve_ivp(
        f, (t0, tf), [delta],
        t_eval=t_eval,
        method="DOP853",   # higher order
        rtol=1e-12,
        atol=1e-14
    )

    if not sol.success:
        print("Solver failed:", sol.message)
    return t_eval, sol

delta = 1e-4
t0 = 0
tf = 2 / delta
h = 0.04
rtol = 1e-4
atol = 1e-6
N = round((tf-t0)/h)
rk_res_t, rk_res_y, n_fevals = rk2(f, t0, tf, delta, h0 = h)
t_eval, sol = libsol(t0, tf, delta, rk_res_t)

tol = rtol * rk_res_y[-1]
print(tol)


# Interpolate adaptive RK solution onto the same N-point grid
rk_res_y_1d = rk_res_y[:, 0]              # shape (m,)
err = np.max(np.abs(sol.y[0] - rk_res_y_1d))

# tol satisfied?
tol_vec = rtol * np.abs(sol.y[0]) + atol
err_vec = np.abs(sol.y[0] - rk_res_y_1d)
satisfied = err_vec < tol_vec
print(np.all(satisfied))

# Plot both on the same grid
plt.figure(figsize=(8, 5))
plt.plot(t_eval, sol.y[0], label="SciPy DOP853")
plt.plot(t_eval, rk_res_y_1d, "--", label="My RK2 interpolated")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(
    rf"$y' = y^2 - y^3$,  $y_0 = {delta}$,  $h_0 = {h}$"
    "\n"
    rf"max_err $= {err:.2e}$,   f_evals $= {n_fevals:,}$"
)
plt.grid(True)
plt.legend()

#plt.show()
plt.savefig("a2/harh/img/rk2_lib_comp.png")

delta = 0.002
tf = 2/delta
hs = [1e0, 1e-1, 1e-2, 1e-3]
errors= []

print(f"{'N':>5} {'h':>10} {'L_inf error':>14}")
print("-" * 65)
for h in hs:
    rk_res_t, rk_res_y, n_steps  = rk2(f, t0, tf, delta, h0 = h)
    N = round((tf-t0)/h)
    rk_res_y_1d = rk_res_y[:, 0]              # shape (m,)

    t_eval, sol = libsol(t0, tf, delta, rk_res_t)

    err = float(np.max(np.abs(sol.y[0] - rk_res_y_1d)))
    errors.append(err)
    print(f"{N:>5} {h:>10.5f} {err:>14.6e}")

hs = np.array(hs)
errors = np.array(errors)

print("\nObserved rate (log2 of error ratio per halving):")
for i in range(1, len(errors)):
    rate = np.log2(errors[i - 1] / errors[i])
    print(f"  N = {hs[i-1]:>4} -> {hs[i]:>4}:   rate = {rate:.3f}")

slope, intercept = np.polyfit(np.log(hs), np.log(errors), 1)
print(f"\nLog-log linear fit slope: {slope:.3f}")

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(7, 5))
ax.loglog(hs, errors, "o-", label="RK2 error")
ref = errors[0] * (hs / hs[0]) ** 2
ax.loglog(hs, ref, "k--", label=r"$\mathcal{O}(h^2)$ reference")
ax.set_xlabel("h")
ax.set_ylabel(r"$\max_i\,|u(x_i, T) - u_i^M|$")
ax.set_title(f"RK2 convergence on interval $[{t0},{tf}]$ with $delta = {delta}$")
ax.legend()
ax.grid(True, which="both", ls=":", alpha=0.6)
fig.tight_layout()

out_png = "a2/harh/img/rk2_conv.png"
#fig.show()
fig.savefig(out_png, dpi=130)
print(f"\nSaved convergence plot to {out_png}")

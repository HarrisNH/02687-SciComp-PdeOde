import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rk3(f, t0, tf, y0, h0=None):
    y = np.atleast_1d(y0).astype(float)
    t = t0

    if h0 is None:
        h = (tf - t0) / 100
    else:
        h = h0

    t_history = [t]
    y_history = [y.copy()]
    
    n_fevals = 0
    while t < tf:
        n_fevals += 3
        if t + h > tf:
            h = tf - t

        k1 = f(t, y)


        k2 = f(t + 1/2 * h, y + 1/2 * h * k1)
        k3 = f(t + h, y -  h * k1 + 2 * h * k2)
        y = y + h * (1/6 * k1 + 2/3 * k2 + 1/6 * k3)

        t = t + h
    
        t_history.append(t)
        y_history.append(y.copy())

    return np.array(t_history), np.array(y_history), n_fevals

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

delta = 0.0001
t0 = 0
tf = 2 / delta
h = 0.314
rtol = 1e-4
atol = 1e-6
N = round((tf-t0)/h)

rk_res_t, rk_res_y, n_fevals = rk3(f, t0, tf, delta, h0 = h)

t_eval, sol = libsol(t0, tf, delta, rk_res_t)


rk_res_y_1d = rk_res_y[:, 0]     
tol = rtol * rk_res_y_1d[-1]       
print(tol)  
err = np.max(np.abs(sol.y[0] - rk_res_y_1d))

# tol satisfied?
tol_vec = rtol * np.abs(sol.y[0]) + atol
err_vec = np.abs(sol.y[0] - rk_res_y_1d)
satisfied = err_vec < tol_vec
print(np.all(satisfied))

# Plot both on the same grid
plt.figure(figsize=(8, 5))
plt.plot(t_eval, sol.y[0], label="SciPy DOP853")
plt.plot(t_eval, rk_res_y_1d, "--", label="RK3")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(
    rf"$y' = y^2 - y^3$,  $y_0 = {delta}$,  $h_0 = {h}$"
    "\n"
    rf"max err $= {err:.2e}$,   f_evals $= {n_fevals:,}$"
)
plt.grid(True)
plt.legend()

#plt.show()
plt.savefig(f"a2/harh/img/rk3_lib_comp_delta_{delta}.png")

delta = 0.002
tf = 2/delta
hs = [1e0, 1e-1, 1e-2]
errors= []

print(f"{'N':>5} {'h':>10} {'L_inf error':>14}")
print("-" * 65)
for h in hs:
    rk_res_t, rk_res_y, n_steps  = rk3(f, t0, tf, delta, h0 = h)
    N = round((tf-t0)/h)
    rk_res_y_1d = rk_res_y[:, 0]              # shape (m,)

    t_eval, sol = libsol(t0, tf, delta, rk_res_t)
    rk_interp = np.interp(t_eval, rk_res_t, rk_res_y_1d)

    err = float(np.max(np.abs(sol.y - rk_interp)))
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
ax.loglog(hs, errors, "o-", label="rk3 error")
ref = errors[0] * (hs / hs[0]) ** 3
ax.loglog(hs, ref, "k--", label=r"$\mathcal{O}(h^3)$ reference")
ax.set_xlabel("h")
ax.set_ylabel(r"$\max_i\,|u(x_i, T) - u_i^M|$")
ax.set_title(f"rk3 convergence on interval $[{t0},{tf}]$, slope = {slope:.3f}")
ax.legend()
ax.grid(True, which="both", ls=":", alpha=0.6)
fig.tight_layout()

out_png = "a2/harh/img/rk3_conv.png"
#fig.show()
fig.savefig(out_png, dpi=130)
print(f"\nSaved convergence plot to {out_png}")

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def update_step_size(h, tol, err, p, safety=0.9, min_factor=0.1, max_factor=5.0):
    factor = safety * (tol/err)**(1/(p+1))
    factor = np.clip(factor, min_factor, max_factor)
    return (h * factor)

def rk23_adaptive(f, t0, tf, y0, rtol=1e-3, atol=1e-6, h0=None):
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
        y_new = y + h * (1/6 * k1 + 2/3 * k2 + 1/6 * k3)
        err = h * np.linalg.norm((1/6*k1 - 1/3*k2 + 1/6*k3))
        tol = rtol * np.linalg.norm(y_new) + atol
        if err < tol:
            t = t + h
            y = y_new
    
            t_history.append(t)
            y_history.append(y.copy())
        h = update_step_size(h, tol, err, 2)
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

delta = 1e-4
t0 = 0
tf = 2 / delta
h = 0.05
rtol = 3.5e-10
atol = 3.7e-11
N = round((tf-t0)/h)


rk_res_t, rk_res_y, n_fevals = rk23_adaptive(f, t0, tf, delta, rtol= rtol, atol = atol, h0 = h)
print(f"steps taken: {n_fevals}")
t_eval, sol = libsol(t0, tf, delta, rk_res_t)



# Interpolate adaptive RK solution onto the same N-point grid
rk_res_y_1d = rk_res_y[:, 0]     
tol = rtol * rk_res_y_1d[-1]       
print(tol)  
err = np.max(np.abs(sol.y[0] - rk_res_y_1d))


# Density plot of step points across t
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(rk_res_t, bins=200, color="steelblue", edgecolor="none")
ax.set_xlabel("t")
ax.set_ylabel("steps per bin")
ax.set_title(rf"Step density — adaptive RK23, $y_0 = {delta}$")
ax.grid(True, axis="y", ls=":", alpha=0.6)
fig.tight_layout()
fig.savefig("a2/harh/img/rk23_step_density.png", dpi=130)
plt.show()

# Plot both on the same grid
plt.figure(figsize=(8, 5))
plt.plot(t_eval, sol.y[0], label="SciPy DOP853")
plt.plot(t_eval, rk_res_y_1d, "--", label="RK23")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(
    rf"$y' = y^2 - y^3$,  $y_0 = {delta}$, "
    "\n"
    rf"max error $= {err:.2e}$,   f_evals $= {n_fevals:,}$"
)
plt.grid(True)
plt.legend()

#plt.show()
plt.savefig("a2/harh/img/rk23_lib_comp_max_err_opt.png")

delta = 0.002
tf = 2/delta
hs = [1e0, 1e-1, 1e-2]
errors= []

print(f"{'N':>5} {'h':>10} {'L_inf error':>14}")
print("-" * 65)
for h in hs:
    rk_res_t, rk_res_y, n_steps  = rk23_adaptive(f, t0, tf, delta, h0 = h)
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
ax.loglog(hs, errors, "o-", label="rk23 error")
ref = errors[0] * (hs / hs[0]) ** 3
ax.loglog(hs, ref, "k--", label=r"$\mathcal{O}(h^3)$ reference")
ax.set_xlabel("h")
ax.set_ylabel(r"$\max_i\,|u(x_i, T) - u_i^M|$")
ax.set_title(f"rk3 convergence on interval $[{t0},{tf}]$ with $delta = {delta}$")
ax.legend()
ax.grid(True, which="both", ls=":", alpha=0.6)
fig.tight_layout()

out_png = "a2/harh/img/rk23_conv.png"
#fig.show()
fig.savefig(out_png, dpi=130)
print(f"\nSaved convergence plot to {out_png}")

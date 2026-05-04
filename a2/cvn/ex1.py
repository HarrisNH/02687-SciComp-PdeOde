import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rk23_adaptive(f, t0, tf, y0, rtol=1e-3, atol=1e-6, h0=None):
    y = np.atleast_1d(y0).astype(float)
    t = t0

    if h0 is None:
        h = (tf - t0) / 100
    else:
        h = h0

    t_history = [t]
    y_history = [y.copy()]

    k_prev = None

    safety = 0.99
    min_factor = 0.2
    max_factor = 5.0

    while t < tf:
        if t + h > tf:
            h = tf - t

        if k_prev is None:
            k1 = f(t, y)
        else:
            k1 = k_prev

        k2 = f(t + h/2, y + h * 0.5 * k1)
        k3 = f(t + 3*h/4, y + h * 0.75 * k2)

        y_new = y + h * (2/9 * k1 + 1/3 * k2 + 4/9 * k3)
        k4 = f(t + h, y_new)

        y_2nd = y + h * (7/24 * k1 + 1/4 * k2 + 1/3 * k3 + 1/8 * k4)

        err_vec = y_new - y_2nd
        tol = rtol * np.abs(y_new) + atol
        err_norm = np.linalg.norm(err_vec / tol)

        if err_norm <= 1.0:
            t = t + h
            y = y_new
            k_prev = k4

            t_history.append(t)
            y_history.append(y.copy())

        if err_norm > 0:
            h_new = h * safety * (1.0 / err_norm)**(1/3)
        else:
            h_new = h * max_factor

        factor = h_new / h
        factor = max(min_factor, min(factor, max_factor))
        h = h * factor

    return np.array(t_history), np.array(y_history)

# Problem setup
N = 100
delta = 0.0001

def f(t, y):
    return y**2 - y**3

t0 = 0
tf = 2 / delta
t_eval = np.linspace(t0, tf, N)

sol = solve_ivp(
    f,
    (t0, tf),
    [delta],
    t_eval=t_eval,
    method="RK23",
    rtol=1e-8,
    atol=1e-10
)

if not sol.success:
    print("Solver failed:", sol.message)


rk_res_t, rk_res_y = rk23_adaptive(f, t0, tf, delta, rtol=1e-3, atol=1e-6)

# Interpolate adaptive RK solution onto the same N-point grid
rk_res_y_1d = rk_res_y[:, 0]              # shape (m,)
rk_interp = np.interp(t_eval, rk_res_t, rk_res_y_1d)

# Plot both on the same grid
plt.figure(figsize=(8, 5))
plt.plot(t_eval, sol.y[0], label="SciPy RK45")
plt.plot(t_eval, rk_interp, "--", label="My RK23 interpolated")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.title(r"Solution of $y' = y^2 - y^3$, $y(0)=\delta$")
plt.grid(True)
plt.legend()
plt.show()
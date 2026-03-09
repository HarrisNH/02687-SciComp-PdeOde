import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags, eye, kron
from scipy.sparse.linalg import cg, LinearOperator
import ex3_a as fc


def lambda_pq(omega, p, q, h):
    ev = (1 - omega) + omega / 2 * (np.cos(p * np.pi * h) + np.cos(q * np.pi * h))
    return ev


def smooth(U, omega, m, F):
    """
    Use this smart update:
    U^(k+1) = u^k + omega * D^-1 * R

    """
    h = 1.0 / (m + 1)
    R = F - fc.Amult(U, m)  # residual
    Unew = U + omega * (h**2 / 4.0) * R
    return Unew


def main():
    omegas = np.linspace(0, 2, 100)
    m_arr = np.array([5, 10, 100, 1000])

    colors = plt.cm.viridis(np.linspace(0, 1, len(m_arr)))
    fig, ax = plt.subplots()
    lamb_arr = np.zeros((len(omegas), len(m_arr)))
    for i, m in enumerate(m_arr):
        q = np.arange(m // 2 + 1, m + 1)
        p = np.arange(m // 2 + 1, m + 1)
        h = 1 / (m + 1)
        P, Q = np.meshgrid(p, q, indexing="ij")
        for j, omega in enumerate(omegas):
            vals = lambda_pq(omega, P, Q, h)
            lamb_arr[j, i] = np.max(np.abs(vals))
        ax.plot(omegas, lamb_arr[:, i], color=colors[i], label=f"m ={m}")
    ax.legend()
    ax.set_title("Max abs high frequency eigenvalues")
    plt.savefig("a1/cvn/ex3_b_max_ev.py")


if __name__ == "__main__":
    main()

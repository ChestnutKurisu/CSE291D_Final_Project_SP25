# Simple 2-D finite-difference simulations for P, S, SH and SV waves
# These snippets replicate the examples from the project documentation.

import numpy as np
import matplotlib.pyplot as plt


def ricker_wavelet(t, f0):
    """Return a Ricker wavelet with dominant frequency ``f0`` at time ``t``."""
    tau = 1.0 / f0
    arg = np.pi * f0 * (t - tau)
    return (1.0 - 2.0 * arg**2) * np.exp(-arg**2)


# ---------------------------------------------------------------------------
# P-WAVE SIMULATION
# ---------------------------------------------------------------------------

def simulate_p_wave(nx=400, nz=400, dx=5.0, dz=5.0, alpha=3000.0, dt=0.0005, nt=750, f0=15.0):
    P_now = np.zeros((nx, nz))
    P_prev = np.zeros((nx, nz))
    P_next = np.zeros((nx, nz))
    sx, sz = nx // 2, nz // 2

    snapshots = []
    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (P_now[i + 1, j] - 2.0 * P_now[i, j] + P_now[i - 1, j]) / dx**2
                    + (P_now[i, j + 1] - 2.0 * P_now[i, j] + P_now[i, j - 1]) / dz**2
                )
                P_next[i, j] = 2.0 * P_now[i, j] - P_prev[i, j] + alpha**2 * dt**2 * lap

        P_next[sx, sz] += ricker_wavelet(it * dt, f0)
        P_prev, P_now, P_next = P_now, P_next, P_prev
        if it % 50 == 0:
            snapshots.append(P_now.copy())
    return snapshots, P_now


# ---------------------------------------------------------------------------
# S-WAVE SIMULATION (scalar potential)
# ---------------------------------------------------------------------------

def simulate_s_wave(nx=400, nz=400, dx=5.0, dz=5.0, beta=1500.0, dt=0.0005, nt=750, f0=15.0):
    S_now = np.zeros((nx, nz))
    S_prev = np.zeros((nx, nz))
    S_next = np.zeros((nx, nz))
    sx, sz = nx // 2, nz // 2

    snapshots = []
    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (S_now[i + 1, j] - 2.0 * S_now[i, j] + S_now[i - 1, j]) / dx**2
                    + (S_now[i, j + 1] - 2.0 * S_now[i, j] + S_now[i, j - 1]) / dz**2
                )
                S_next[i, j] = 2.0 * S_now[i, j] - S_prev[i, j] + beta**2 * dt**2 * lap

        S_next[sx, sz] += ricker_wavelet(it * dt, f0)
        S_prev, S_now, S_next = S_now, S_next, S_prev
        if it % 50 == 0:
            snapshots.append(S_now.copy())
    return snapshots, S_now


# ---------------------------------------------------------------------------
# SH-WAVE SIMULATION
# ---------------------------------------------------------------------------

def simulate_sh_wave(nx=400, nz=400, dx=5.0, dz=5.0, beta=1500.0, dt=0.0005, nt=750, f0=15.0):
    uY_now = np.zeros((nx, nz))
    uY_prev = np.zeros((nx, nz))
    uY_next = np.zeros((nx, nz))
    sx, sz = nx // 2, nz // 2

    snapshots = []
    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (uY_now[i + 1, j] - 2.0 * uY_now[i, j] + uY_now[i - 1, j]) / dx**2
                    + (uY_now[i, j + 1] - 2.0 * uY_now[i, j] + uY_now[i, j - 1]) / dz**2
                )
                uY_next[i, j] = 2.0 * uY_now[i, j] - uY_prev[i, j] + beta**2 * dt**2 * lap

        uY_next[sx, sz] += ricker_wavelet(it * dt, f0)
        uY_prev, uY_now, uY_next = uY_now, uY_next, uY_prev
        if it % 50 == 0:
            snapshots.append(uY_now.copy())
    return snapshots, uY_now


# ---------------------------------------------------------------------------
# SV-WAVE SIMULATION (scalar potential)
# ---------------------------------------------------------------------------

def simulate_sv_wave(nx=400, nz=400, dx=5.0, dz=5.0, beta=1500.0, dt=0.0005, nt=750, f0=15.0):
    Psi_now = np.zeros((nx, nz))
    Psi_prev = np.zeros((nx, nz))
    Psi_next = np.zeros((nx, nz))
    sx, sz = nx // 2, nz // 2

    snapshots = []
    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (Psi_now[i + 1, j] - 2.0 * Psi_now[i, j] + Psi_now[i - 1, j]) / dx**2
                    + (Psi_now[i, j + 1] - 2.0 * Psi_now[i, j] + Psi_now[i, j - 1]) / dz**2
                )
                Psi_next[i, j] = 2.0 * Psi_now[i, j] - Psi_prev[i, j] + beta**2 * dt**2 * lap

        Psi_next[sx, sz] += ricker_wavelet(it * dt, f0)
        Psi_prev, Psi_now, Psi_next = Psi_now, Psi_next, Psi_prev
        if it % 50 == 0:
            snapshots.append(Psi_now.copy())
    return snapshots, Psi_now


if __name__ == "__main__":
    snaps, field = simulate_p_wave(nt=400)
    plt.imshow(field.T, cmap="seismic", origin="lower", aspect="auto")
    plt.title("Final P-Wave Field")
    plt.colorbar(label="Amplitude")
    plt.show()

import numpy as np


def ricker_wavelet(t: float, f0: float) -> float:
    """Return Ricker wavelet value at time ``t`` for peak frequency ``f0``."""
    tau = 1.0 / f0
    arg = np.pi * f0 * (t - tau)
    return (1.0 - 2.0 * arg ** 2) * np.exp(-arg ** 2)


def solve_p_wave(nx: int = 200, nz: int = 200, dx: float = 5.0, dz: float = 5.0,
                 alpha: float = 3000.0, dt: float = 0.0005, nt: int = 750,
                 f0: float = 15.0):
    """Compute a 2-D P-wave field using a simple finite difference solver.

    Parameters
    ----------
    nx, nz : int
        Number of grid points in the x and z directions.
    dx, dz : float
        Grid spacing in meters.
    alpha : float
        P-wave speed.
    dt : float
        Time step in seconds.
    nt : int
        Number of time steps to simulate.
    f0 : float
        Peak frequency of the Ricker wavelet source.

    Returns
    -------
    list[np.ndarray]
        Snapshot wave fields taken every 50 steps.
    np.ndarray
        Final wave field array of shape ``(nx, nz)``.
    """
    P_now = np.zeros((nx, nz))
    P_prev = np.zeros((nx, nz))
    P_next = np.zeros((nx, nz))

    sx, sz = nx // 2, nz // 2

    snapshots = []

    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (P_now[i + 1, j] - 2.0 * P_now[i, j] + P_now[i - 1, j]) / dx ** 2
                    + (P_now[i, j + 1] - 2.0 * P_now[i, j] + P_now[i, j - 1]) / dz ** 2
                )
                P_next[i, j] = (
                    2.0 * P_now[i, j] - P_prev[i, j] + alpha ** 2 * dt ** 2 * lap
                )

        P_next[sx, sz] += ricker_wavelet(it * dt, f0)

        P_prev, P_now, P_next = P_now, P_next, P_prev

        if it % 50 == 0:
            snapshots.append(P_now.copy())

    return snapshots, P_now


def solve_s_wave(nx: int = 200, nz: int = 200, dx: float = 5.0, dz: float = 5.0,
                 beta: float = 1500.0, dt: float = 0.0005, nt: int = 750,
                 f0: float = 15.0):
    """Compute a 2-D S-wave field using a simple finite difference solver."""
    S_now = np.zeros((nx, nz))
    S_prev = np.zeros((nx, nz))
    S_next = np.zeros((nx, nz))

    sx, sz = nx // 2, nz // 2
    snapshots = []

    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (S_now[i + 1, j] - 2.0 * S_now[i, j] + S_now[i - 1, j]) / dx ** 2
                    + (S_now[i, j + 1] - 2.0 * S_now[i, j] + S_now[i, j - 1]) / dz ** 2
                )
                S_next[i, j] = (
                    2.0 * S_now[i, j] - S_prev[i, j] + beta ** 2 * dt ** 2 * lap
                )

        S_next[sx, sz] += ricker_wavelet(it * dt, f0)

        S_prev, S_now, S_next = S_now, S_next, S_prev

        if it % 50 == 0:
            snapshots.append(S_now.copy())

    return snapshots, S_now

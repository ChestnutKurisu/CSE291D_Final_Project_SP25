"""P-wave and S-wave finite difference solvers.

This module implements simple 2-D finite difference solvers for acoustic
P-waves and shear S-waves in a homogeneous medium.  The implementation is
kept deliberately straightforward and mirrors the numerical formulations in
``simulation.py`` so that it can serve as a clear reference implementation.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# utility
# ---------------------------------------------------------------------------
def ricker_wavelet(t: float, f0: float) -> float:
    """Return the value of a Ricker wavelet at time ``t`` for peak ``f0``."""
    tau = 1.0 / f0
    term = np.pi * f0 * (t - tau)
    return (1.0 - 2.0 * term ** 2) * np.exp(-term ** 2)


# ---------------------------------------------------------------------------
# P-wave solver
# ---------------------------------------------------------------------------
def simulate_p_wave(
    nx: int = 200,
    nz: int = 200,
    dx: float = 5.0,
    dz: float = 5.0,
    alpha: float = 3000.0,
    dt: float = 5e-4,
    nt: int = 750,
    f0: float = 15.0,
):
    """Simulate a 2‑D acoustic P-wave field.

    Parameters
    ----------
    nx, nz : int
        Number of grid points in the ``x`` and ``z`` directions.
    dx, dz : float
        Grid spacing (metres).
    alpha : float
        P-wave speed.
    dt : float
        Time step size.
    nt : int
        Number of time steps.
    f0 : float
        Peak frequency of the Ricker source wavelet.

    Returns
    -------
    np.ndarray
        The final wavefield.
    list[np.ndarray]
        Snapshots of the wavefield every 50 steps.
    """
    p_now = np.zeros((nx, nz), dtype=float)
    p_prev = np.zeros_like(p_now)
    p_next = np.zeros_like(p_now)

    sx, sz = nx // 2, nz // 2
    snaps: list[np.ndarray] = []

    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (p_now[i + 1, j] - 2.0 * p_now[i, j] + p_now[i - 1, j]) / dx ** 2
                    + (
                        p_now[i, j + 1]
                        - 2.0 * p_now[i, j]
                        + p_now[i, j - 1]
                    )
                    / dz ** 2
                )
                p_next[i, j] = (
                    2.0 * p_now[i, j]
                    - p_prev[i, j]
                    + alpha ** 2 * dt ** 2 * lap
                )

        p_next[sx, sz] += ricker_wavelet(it * dt, f0)
        p_prev, p_now, p_next = p_now, p_next, p_prev
        if it % 50 == 0:
            snaps.append(p_now.copy())

    return p_now, snaps


# ---------------------------------------------------------------------------
# S-wave solver
# ---------------------------------------------------------------------------
def simulate_s_wave(
    nx: int = 200,
    nz: int = 200,
    dx: float = 5.0,
    dz: float = 5.0,
    beta: float = 1500.0,
    dt: float = 5e-4,
    nt: int = 750,
    f0: float = 15.0,
):
    """Simulate a 2‑D shear S-wave field.

    The structure mirrors :func:`simulate_p_wave` but uses the shear speed
    ``beta`` instead of the P-wave speed ``alpha``.
    """
    s_now = np.zeros((nx, nz), dtype=float)
    s_prev = np.zeros_like(s_now)
    s_next = np.zeros_like(s_now)

    sx, sz = nx // 2, nz // 2
    snaps: list[np.ndarray] = []

    for it in range(nt):
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap = (
                    (s_now[i + 1, j] - 2.0 * s_now[i, j] + s_now[i - 1, j]) / dx ** 2
                    + (
                        s_now[i, j + 1]
                        - 2.0 * s_now[i, j]
                        + s_now[i, j - 1]
                    )
                    / dz ** 2
                )
                s_next[i, j] = (
                    2.0 * s_now[i, j]
                    - s_prev[i, j]
                    + beta ** 2 * dt ** 2 * lap
                )

        s_next[sx, sz] += ricker_wavelet(it * dt, f0)
        s_prev, s_now, s_next = s_now, s_next, s_prev
        if it % 50 == 0:
            snaps.append(s_now.copy())

    return s_now, snaps

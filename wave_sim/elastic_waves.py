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

# ---------------------------------------------------------------------------
# Full elastic solver using P and S wave potentials
# ---------------------------------------------------------------------------
def simulate_elastic_potentials(
    nx: int = 200,
    nz: int = 200,
    dx: float = 5.0,
    dz: float = 5.0,
    vp: float = 3000.0,
    vs: float = 1500.0,
    dt: float | None = None,
    nt: int = 750,
    f0: float = 15.0,
    source: str = "P",
    absorb_width: int = 10,
    absorb_strength: float = 2.0,
):
    """Simulate 2-D elastic wave propagation via scalar potentials.

    The solver evolves a P-wave potential ``phi`` and an S-wave potential ``psi``
    according to the coupled potential formulation of isotropic elasticity::

        phi_tt = vp^2 * (phi_xx + phi_zz)
        psi_tt = vs^2 * (psi_xx + psi_zz)

    Displacements ``u_x`` and ``u_z`` are recovered from the potentials via

        u_x = dphi_dx - dpsi_dz
        u_z = dphi_dz + dpsi_dx

    Parameters
    ----------
    nx, nz : int
        Number of grid points in the ``x`` and ``z`` directions.
    dx, dz : float
        Grid spacing (metres).
    vp, vs : float
        P-wave and S-wave speeds.
    dt : float, optional
        Time step size.  If ``None`` a stable value based on ``vp`` is chosen.
    nt : int
        Number of time steps to simulate.
    f0 : float
        Peak frequency of the Ricker source wavelet.
    source : {'P', 'S', 'both'}
        Which potential receives the source term.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Final displacement fields ``(u_x, u_z)``.
    list[tuple[np.ndarray, np.ndarray]]
        Displacement snapshots every 50 steps.
    """
    if dt is None:
        cfl = 0.7
        dt = cfl * min(dx, dz) / max(vp, vs)

    phi_now = np.zeros((nx, nz), dtype=float)
    phi_prev = np.zeros_like(phi_now)
    phi_next = np.zeros_like(phi_now)

    psi_now = np.zeros((nx, nz), dtype=float)
    psi_prev = np.zeros_like(psi_now)
    psi_next = np.zeros_like(psi_now)

    damping = np.ones((nx, nz))
    for i in range(absorb_width):
        coef = np.exp(-absorb_strength * ((absorb_width - i) / absorb_width) ** 2)
        damping[i, :] *= coef
        damping[-i - 1, :] *= coef
        damping[:, i] *= coef
        damping[:, -i - 1] *= coef

    ux = np.zeros((nx, nz), dtype=float)
    uz = np.zeros((nx, nz), dtype=float)

    sx, sz = nx // 2, nz // 2
    snaps: list[tuple[np.ndarray, np.ndarray]] = []

    vp2_dt2 = (vp * dt) ** 2
    vs2_dt2 = (vs * dt) ** 2

    for it in range(nt):
        # update phi potential
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap_phi = (
                    (phi_now[i + 1, j] - 2.0 * phi_now[i, j] + phi_now[i - 1, j]) / dx ** 2
                    + (
                        phi_now[i, j + 1]
                        - 2.0 * phi_now[i, j]
                        + phi_now[i, j - 1]
                    )
                    / dz ** 2
                )
                phi_next[i, j] = 2.0 * phi_now[i, j] - phi_prev[i, j] + vp2_dt2 * lap_phi

        # update psi potential
        for i in range(1, nx - 1):
            for j in range(1, nz - 1):
                lap_psi = (
                    (psi_now[i + 1, j] - 2.0 * psi_now[i, j] + psi_now[i - 1, j]) / dx ** 2
                    + (
                        psi_now[i, j + 1]
                        - 2.0 * psi_now[i, j]
                        + psi_now[i, j - 1]
                    )
                    / dz ** 2
                )
                psi_next[i, j] = 2.0 * psi_now[i, j] - psi_prev[i, j] + vs2_dt2 * lap_psi

        src_val = ricker_wavelet(it * dt, f0)
        if source in ("P", "both"):
            phi_next[sx, sz] += src_val
        if source in ("S", "both"):
            psi_next[sx, sz] += src_val

        phi_next *= damping
        psi_next *= damping

        phi_prev, phi_now, phi_next = phi_now, phi_next, phi_prev
        psi_prev, psi_now, psi_next = psi_now, psi_next, psi_prev

        if it % 50 == 0:
            # compute displacements from current potentials
            dphi_dx = (phi_now[2:, 1:-1] - phi_now[:-2, 1:-1]) / (2 * dx)
            dphi_dz = (phi_now[1:-1, 2:] - phi_now[1:-1, :-2]) / (2 * dz)
            dpsi_dx = (psi_now[2:, 1:-1] - psi_now[:-2, 1:-1]) / (2 * dx)
            dpsi_dz = (psi_now[1:-1, 2:] - psi_now[1:-1, :-2]) / (2 * dz)

            ux[1:-1, 1:-1] = dphi_dx - dpsi_dz
            uz[1:-1, 1:-1] = dphi_dz + dpsi_dx
            snaps.append((ux.copy(), uz.copy()))

    # final displacement
    dphi_dx = (phi_now[2:, 1:-1] - phi_now[:-2, 1:-1]) / (2 * dx)
    dphi_dz = (phi_now[1:-1, 2:] - phi_now[1:-1, :-2]) / (2 * dz)
    dpsi_dx = (psi_now[2:, 1:-1] - psi_now[:-2, 1:-1]) / (2 * dx)
    dpsi_dz = (psi_now[1:-1, 2:] - psi_now[1:-1, :-2]) / (2 * dz)

    ux[1:-1, 1:-1] = dphi_dx - dpsi_dz
    uz[1:-1, 1:-1] = dphi_dz + dpsi_dx

    return (ux, uz), snaps

# -*- coding: utf-8 -*-
"""Standalone 1-D wave solvers used in the examples.

These classes implement simple finite-difference or spectral
schemes for various wave phenomena.  They are primarily intended
for demonstration and are not optimised for large problems.
"""

from __future__ import annotations

import numpy as np


class PlaneAcousticWave:
    """Solve the one-dimensional acoustic wave equation."""

    def __init__(self, c: float = 1.0, L: float = 1.0, Nx: int = 200, dt: float = 0.001, T: float = 1.0) -> None:
        self.c = c
        self.L = L
        self.Nx = Nx
        self.dx = L / Nx
        self.dt = dt
        self.T = T

        self.x = np.linspace(0, L, Nx + 1)
        self.nt = int(T // dt)

        self.p_now = np.zeros(Nx + 1)
        self.p_prev = np.zeros(Nx + 1)
        self.p_next = np.zeros(Nx + 1)

    def initial_conditions(self, p_init_func, dp_init_func=None) -> None:
        for i in range(self.Nx + 1):
            self.p_now[i] = p_init_func(self.x[i])

        if dp_init_func is not None:
            for i in range(self.Nx + 1):
                self.p_prev[i] = self.p_now[i] - self.dt * dp_init_func(self.x[i])
        else:
            self.p_prev[:] = self.p_now[:]

    def apply_boundaries(self, arr: np.ndarray) -> None:
        arr[0] = 0.0
        arr[-1] = 0.0

    def step(self) -> None:
        c2 = self.c ** 2
        r = c2 * (self.dt ** 2 / self.dx ** 2)
        for i in range(1, self.Nx):
            self.p_next[i] = (
                2.0 * self.p_now[i]
                - self.p_prev[i]
                + r * (self.p_now[i + 1] - 2.0 * self.p_now[i] + self.p_now[i - 1])
            )
        self.apply_boundaries(self.p_next)
        self.p_prev, self.p_now, self.p_next = self.p_now, self.p_next, self.p_prev

    def solve(self) -> np.ndarray:
        psol = [self.p_now.copy()]
        for _ in range(self.nt):
            self.step()
            psol.append(self.p_now.copy())
        return np.array(psol)


class SphericalAcousticWave:
    """Solve the spherically symmetric acoustic wave equation."""

    def __init__(self, c: float = 1.0, R: float = 2.0, Nr: int = 200, dt: float = 0.001, T: float = 1.0) -> None:
        self.c = c
        self.R = R
        self.Nr = Nr
        self.dr = R / Nr
        self.dt = dt
        self.T = T

        self.r = np.linspace(0, R, Nr + 1)
        self.nt = int(T // dt)

        self.p_now = np.zeros(Nr + 1)
        self.p_prev = np.zeros(Nr + 1)
        self.p_next = np.zeros(Nr + 1)

    def initial_conditions(self, p_init_func, dp_init_func=None) -> None:
        for i in range(self.Nr + 1):
            self.p_now[i] = p_init_func(self.r[i])
        if dp_init_func is not None:
            for i in range(self.Nr + 1):
                self.p_prev[i] = self.p_now[i] - self.dt * dp_init_func(self.r[i])
        else:
            self.p_prev[:] = self.p_now[:]

    def apply_boundary_conditions(self, arr: np.ndarray) -> None:
        arr[0] = arr[1]
        arr[-1] = 0.0

    def step(self) -> None:
        rfac = self.c ** 2 * self.dt ** 2 / (self.dr ** 2)
        for i in range(1, self.Nr):
            r_i = self.r[i]
            dpdr_plus = self.p_now[i + 1] - self.p_now[i]
            dpdr_minus = self.p_now[i] - self.p_now[i - 1]
            term_plus = ((i + 0.5) * self.dr) ** 2 * dpdr_plus
            term_minus = ((i - 0.5) * self.dr) ** 2 * dpdr_minus
            self.p_next[i] = (
                2.0 * self.p_now[i]
                - self.p_prev[i]
                + rfac * (term_plus - term_minus) / ((r_i ** 2) * self.dr)
            )
        self.apply_boundary_conditions(self.p_next)
        self.p_prev, self.p_now, self.p_next = self.p_now, self.p_next, self.p_prev

    def solve(self) -> np.ndarray:
        sol = [self.p_now.copy()]
        for _ in range(self.nt):
            self.step()
            sol.append(self.p_now.copy())
        return np.array(sol)


class DeepWaterGravityWave:
    """Spectral solver for 1-D deep-water gravity waves."""

    def __init__(self, L: float = 2 * np.pi, Nx: int = 256, g: float = 9.81, dt: float = 0.01, T: float = 2.0) -> None:
        self.L = L
        self.Nx = Nx
        self.dx = L / Nx
        self.x = np.linspace(0, L - self.dx, Nx)
        self.g = g
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)

        self.k = np.fft.rfftfreq(self.Nx, d=self.dx) * 2 * np.pi
        self.eta_hat_now = np.zeros_like(self.k, dtype=np.complex128)
        self.eta_hat_prev = np.zeros_like(self.k, dtype=np.complex128)
        self.eta_now = np.zeros(self.Nx)

    def initial_conditions(self, eta_init_func, deta_init_func=None) -> None:
        self.eta_now = eta_init_func(self.x)
        self.eta_hat_now = np.fft.rfft(self.eta_now)
        if deta_init_func is not None:
            eta_t_init = deta_init_func(self.x)
            eta_hat_t_init = np.fft.rfft(eta_t_init)
            self.eta_hat_prev = self.eta_hat_now - self.dt * eta_hat_t_init
        else:
            self.eta_hat_prev = self.eta_hat_now.copy()

    def step(self) -> None:
        eta_hat_next = np.zeros_like(self.eta_hat_now, dtype=np.complex128)
        for i in range(len(self.k)):
            k_abs = abs(self.k[i])
            alpha = self.g * k_abs * (self.dt ** 2)
            eta_hat_next[i] = (2.0 - alpha) * self.eta_hat_now[i] - self.eta_hat_prev[i]
        self.eta_hat_prev = self.eta_hat_now
        self.eta_hat_now = eta_hat_next
        self.eta_now = np.fft.irfft(self.eta_hat_now, n=self.Nx)

    def solve(self) -> np.ndarray:
        snapshots = [self.eta_now.copy()]
        for _ in range(self.nt):
            self.step()
            snapshots.append(self.eta_now.copy())
        return np.array(snapshots)


class ShallowWaterGravityWave:
    """Rusanov finite-volume solver for 1-D shallow water equations."""

    def __init__(self, g: float = 9.81, L: float = 10.0, Nx: int = 200, dt: float = 0.001, T: float = 2.0) -> None:
        self.g = g
        self.L = L
        self.Nx = Nx
        self.dx = L / Nx
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)
        self.x = np.linspace(0.5 * self.dx, L - 0.5 * self.dx, Nx)
        self.Q = np.zeros((2, Nx))

    def initial_conditions(self, h_init, u_init) -> None:
        for i in range(self.Nx):
            xi = self.x[i]
            h = h_init(xi)
            u = u_init(xi)
            self.Q[0, i] = h
            self.Q[1, i] = h * u

    def flux(self, Q: np.ndarray) -> np.ndarray:
        h = Q[0]
        hu = Q[1]
        u = np.zeros_like(h)
        mask = h > 1e-12
        u[mask] = hu[mask] / h[mask]
        return np.array([hu, hu * u + 0.5 * self.g * h * h])

    def max_wave_speed(self, Q_left, Q_right) -> float:
        hL, huL = Q_left
        hR, huR = Q_right
        uL = huL / hL if hL > 1e-12 else 0.0
        uR = huR / hR if hR > 1e-12 else 0.0
        cL = np.sqrt(self.g * hL)
        cR = np.sqrt(self.g * hR)
        return max(abs(uL) + cL, abs(uR) + cR)

    def step(self) -> None:
        Qn = self.Q.copy()
        F = self.flux(Qn)
        Fnum = np.zeros_like(F)
        for i in range(self.Nx - 1):
            QL = Qn[:, i]
            QR = Qn[:, i + 1]
            FL = F[:, i]
            FR = F[:, i + 1]
            smax = self.max_wave_speed(QL, QR)
            Fnum[:, i] = 0.5 * (FL + FR) - 0.5 * smax * (QR - QL)
        for i in range(1, self.Nx - 1):
            self.Q[:, i] = Qn[:, i] - (self.dt / self.dx) * (Fnum[:, i] - Fnum[:, i - 1])
        self.Q[:, 0] = Qn[:, 0] - (self.dt / self.dx) * (Fnum[:, 0] - Fnum[:, 0])
        self.Q[:, -1] = Qn[:, -1] - (self.dt / self.dx) * (
            Fnum[:, self.Nx - 2] - Fnum[:, self.Nx - 2]
        )

    def solve(self) -> list[np.ndarray]:
        snapshots = [self.Q.copy()]
        for _ in range(self.nt):
            self.step()
            snapshots.append(self.Q.copy())
        return snapshots


class CapillaryWave:
    """Finite-difference solver for linear capillary waves."""

    def __init__(self, L: float = 2 * np.pi, Nx: int = 256, sigma: float = 0.074, rho: float = 1000.0, dt: float = 0.0001, T: float = 0.1) -> None:
        self.L = L
        self.Nx = Nx
        self.dx = L / Nx
        self.sigma = sigma
        self.rho = rho
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)
        self.x = np.linspace(0, L - self.dx, Nx)
        self.eta_now = np.zeros(Nx)
        self.eta_prev = np.zeros(Nx)
        self.eta_next = np.zeros(Nx)

    def initial_conditions(self, eta_init_func, deta_init_func=None) -> None:
        self.eta_now = eta_init_func(self.x)
        if deta_init_func is not None:
            self.eta_prev = self.eta_now - self.dt * deta_init_func(self.x)
        else:
            self.eta_prev = self.eta_now.copy()

    def periodic_idx(self, i: int) -> int:
        return i % self.Nx

    def fourth_deriv(self, eta_arr: np.ndarray, i: int) -> float:
        return (
            eta_arr[self.periodic_idx(i + 2)]
            - 4 * eta_arr[self.periodic_idx(i + 1)]
            + 6 * eta_arr[self.periodic_idx(i)]
            - 4 * eta_arr[self.periodic_idx(i - 1)]
            + eta_arr[self.periodic_idx(i - 2)]
        )

    def step(self) -> None:
        coef = (self.sigma / self.rho) * (self.dt ** 2) / (self.dx ** 4)
        for i in range(self.Nx):
            d4 = self.fourth_deriv(self.eta_now, i)
            self.eta_next[i] = 2.0 * self.eta_now[i] - self.eta_prev[i] - coef * d4
        self.eta_prev, self.eta_now, self.eta_next = self.eta_now, self.eta_next, self.eta_prev

    def solve(self) -> np.ndarray:
        snapshots = [self.eta_now.copy()]
        for _ in range(self.nt):
            self.step()
            snapshots.append(self.eta_now.copy())
        return np.array(snapshots)


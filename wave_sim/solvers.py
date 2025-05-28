# -*- coding: utf-8 -*-
"""Unified collection of wave solver classes.

This module consolidates the various solver implementations scattered across
``wave_sim`` into a single location.  The class definitions are unchanged from
those originally defined in :mod:`p_wave`, :mod:`s_wave` and
:mod:`basic_wave_solvers`.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .base import WaveSimulation


class PWaveSimulation(WaveSimulation):
    """Finite-difference propagator for a 2-D P-wave field."""

    def __init__(self, f0=15.0, source_pos=None, source_func=None, **kwargs):
        kwargs.setdefault("c", 3000.0)
        kwargs.setdefault("dx", 5.0)
        kwargs.setdefault("dt", 0.0005)
        kwargs.setdefault("backend", "gpu")
        super().__init__(**kwargs)
        self.f0 = f0
        if source_pos is None:
            source_pos = (self.n // 2, self.n // 2)
        self.source_pos = source_pos
        self.initialize(amplitude=0.0, source_func=source_func)

    @staticmethod
    def ricker_wavelet(t, f0):
        tau = 1.0 / f0
        return (1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t - tau) ** 2) * np.exp(
            -(np.pi ** 2) * (f0 ** 2) * (t - tau) ** 2
        )

    def step(self):
        c2 = (self.c * self.dt / self.dx) ** 2
        laplacian = (
            np.roll(self.u_curr, 1, axis=0)
            + np.roll(self.u_curr, -1, axis=0)
            + np.roll(self.u_curr, 1, axis=1)
            + np.roll(self.u_curr, -1, axis=1)
            - 4 * self.u_curr
        )
        u_next = 2 * self.u_curr - self.u_prev + c2 * laplacian

        amp = self.ricker_wavelet(self.time, self.f0)
        sx, sy = self.source_pos
        u_next[sx, sy] += amp

        if self.boundary == "reflective":
            u_next[0, :] = 0
            u_next[-1, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0
        elif self.boundary == "periodic":
            pass
        elif self.boundary == "absorbing":
            u_next[0, :] = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0] = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
        else:
            raise ValueError(f"Unknown boundary condition {self.boundary}")

        self.u_prev, self.u_curr = self.u_curr, u_next
        self.time += self.dt
        return u_next


class SWaveSimulation(WaveSimulation):
    """Finite-difference solver for shear (S) waves."""

    def __init__(self, f0=15.0, source_pos=None, source_func=None, **kwargs):
        kwargs.setdefault("c", 1500.0)
        kwargs.setdefault("dx", 5.0)
        kwargs.setdefault("dt", 0.0005)
        kwargs.setdefault("backend", "gpu")
        super().__init__(**kwargs)
        self.f0 = f0
        if source_pos is None:
            source_pos = (self.n // 2, self.n // 2)
        self.source_pos = source_pos
        self.initialize(amplitude=0.0, source_func=source_func)

    @staticmethod
    def ricker_wavelet(t, f0):
        tau = 1.0 / f0
        return (1.0 - 2.0 * (np.pi ** 2) * (f0 ** 2) * (t - tau) ** 2) * np.exp(
            -(np.pi ** 2) * (f0 ** 2) * (t - tau) ** 2
        )

    def step(self):
        c2 = (self.c * self.dt / self.dx) ** 2
        laplacian = (
            np.roll(self.u_curr, 1, axis=0)
            + np.roll(self.u_curr, -1, axis=0)
            + np.roll(self.u_curr, 1, axis=1)
            + np.roll(self.u_curr, -1, axis=1)
            - 4 * self.u_curr
        )
        u_next = 2 * self.u_curr - self.u_prev + c2 * laplacian

        amp = self.ricker_wavelet(self.time, self.f0)
        sx, sy = self.source_pos
        u_next[sx, sy] += amp

        if self.boundary == "reflective":
            u_next[0, :] = 0
            u_next[-1, :] = 0
            u_next[:, 0] = 0
            u_next[:, -1] = 0
        elif self.boundary == "periodic":
            pass
        elif self.boundary == "absorbing":
            u_next[0, :] = u_next[1, :]
            u_next[-1, :] = u_next[-2, :]
            u_next[:, 0] = u_next[:, 1]
            u_next[:, -1] = u_next[:, -2]
        else:
            raise ValueError(f"Unknown boundary condition {self.boundary}")

        self.u_prev, self.u_curr = self.u_curr, u_next
        self.time += self.dt
        return u_next


class SHWaveSimulation(SWaveSimulation):
    """Horizontally polarised shear wave.

    The scalar field ``u_curr`` directly corresponds to the out-of-plane
    displacement :math:`u_y`.  This subclass simply exposes a helper method to
    access that quantity for consistency with :class:`SVWaveSimulation`.
    """

    def displacement(self) -> np.ndarray:
        """Return the out-of-plane displacement ``u_y``."""

        return self.u_curr


class SVWaveSimulation(SWaveSimulation):
    """Vertically polarised shear wave represented by a scalar potential.

    Physical displacement components ``(u_x, u_z)`` can be recovered from the
    scalar potential :math:`\Psi` via

    .. math::

       u_x = \frac{\partial \Psi}{\partial z},\qquad
       u_z = -\frac{\partial \Psi}{\partial x}.
    """

    def displacement_components(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the displacement components ``(u_x, u_z)``.

        The derivatives are computed with second-order centred differences via
        :func:`numpy.gradient`.  For reflective boundaries the displacement is
        forced to zero at the edges to remain consistent with the field
        boundary condition.
        """

        dpsi_dx, dpsi_dz = np.gradient(self.u_curr, self.dx, self.dx, edge_order=2)
        ux = dpsi_dz
        uz = -dpsi_dx

        if self.boundary == "reflective":
            ux[0, :] = ux[-1, :] = 0.0
            ux[:, 0] = ux[:, -1] = 0.0
            uz[0, :] = uz[-1, :] = 0.0
            uz[:, 0] = uz[:, -1] = 0.0

        return ux, uz


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


class InternalGravityWave:
    """Explicit solver for a 1-D internal gravity wave.

    The scheme integrates ``psi_tt = N**2 * psi_xx`` using centred
    differences in space and time.
    """

    def __init__(self, N: float = 1.0, L: float = 2.0, Nx: int = 800, dt: Optional[float] = None, T: float = 1.0) -> None:
        c = N
        self.N = N
        self.L = L
        self.Nx = Nx
        self.dx = L / Nx
        if dt is None:
            dt = 0.8 * self.dx / c
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)

        self.x = np.linspace(0, L, Nx + 1)
        self.psi_now = np.zeros(Nx + 1)
        self.psi_prev = np.zeros(Nx + 1)
        self.psi_next = np.zeros(Nx + 1)

    def initial_conditions(self, psi_init_func, dpsi_init_func=None) -> None:
        for i in range(self.Nx + 1):
            self.psi_now[i] = psi_init_func(self.x[i])
        if dpsi_init_func is not None:
            for i in range(self.Nx + 1):
                self.psi_prev[i] = self.psi_now[i] - self.dt * dpsi_init_func(self.x[i])
        else:
            self.psi_prev[:] = self.psi_now[:]

    def step(self) -> None:
        c2 = self.N ** 2
        r = c2 * (self.dt ** 2 / self.dx ** 2)
        for i in range(1, self.Nx):
            self.psi_next[i] = (
                2.0 * self.psi_now[i]
                - self.psi_prev[i]
                + r * (self.psi_now[i + 1] - 2.0 * self.psi_now[i] + self.psi_now[i - 1])
            )
        self.psi_next[0] = 0.0
        self.psi_next[-1] = 0.0
        self.psi_prev, self.psi_now, self.psi_next = self.psi_now, self.psi_next, self.psi_prev

    def solve(self) -> np.ndarray:
        sol = [self.psi_now.copy()]
        for _ in range(self.nt):
            self.step()
            sol.append(self.psi_now.copy())
        return np.array(sol)


class KelvinWave:
    """Kelvin wave in the rotating shallow-water system.

    Solves the linearised equations

    ``u_t - f v = -g eta_y``
    ``v_t + f u = 0``
    ``eta_t + H u_y = 0``

    with centred finite differences along the ``y`` direction.
    """

    def __init__(self, L: float = 10.0, Ny: int = 800, H: float = 1.0, f: float = 1.0, g: float = 9.81, dt: Optional[float] = None, T: float = 10.0) -> None:
        self.L = L
        self.Ny = Ny
        self.dy = L / (Ny - 1)
        self.H = H
        self.f = f
        self.g = g
        if dt is None:
            c = np.sqrt(g * H)
            dt = 0.4 * self.dy / c
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)

        self.y = np.linspace(0, L, Ny)
        self.u = np.zeros(Ny)
        self.v = np.zeros(Ny)
        self.eta = np.zeros(Ny)

    def initial_conditions(self, eta_init_func) -> None:
        self.eta = eta_init_func(self.y)

    def step(self) -> None:
        u_new = self.u.copy()
        v_new = self.v.copy()
        eta_new = self.eta.copy()
        for j in range(1, self.Ny - 1):
            du = -self.g * (self.eta[j + 1] - self.eta[j - 1]) / (2 * self.dy) + self.f * self.v[j]
            dv = -self.f * self.u[j]
            deta = -self.H * (self.u[j + 1] - self.u[j - 1]) / (2 * self.dy)
            u_new[j] = self.u[j] + self.dt * du
            v_new[j] = self.v[j] + self.dt * dv
            eta_new[j] = self.eta[j] + self.dt * deta
        u_new[0] = 0.0
        v_new[0] = 0.0
        eta_new[0] = self.eta[0]
        u_new[-1] = self.u[-1]
        v_new[-1] = self.v[-1]
        eta_new[-1] = self.eta[-1]
        self.u, self.v, self.eta = u_new, v_new, eta_new

    def solve(self) -> np.ndarray:
        snaps = [self.eta.copy()]
        for _ in range(self.nt):
            self.step()
            snaps.append(self.eta.copy())
        return np.array(snaps)


class RossbyPlanetaryWave:
    """Spectral solver for the linear barotropic vorticity equation.

    Integrates ``(∇²ψ)_t + β ψ_x = 0`` on a periodic domain using FFTs.
    """

    def __init__(self, Nx: int = 256, Ny: int = 256, Lx: float = 2 * np.pi, Ly: float = 2 * np.pi, beta: float = 1.0, dt: float = 0.01, T: float = 2.0) -> None:
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / Nx
        self.dy = Ly / Ny
        self.beta = beta
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)

        self.kx = np.fft.fftfreq(Nx, d=self.dx) * 2 * np.pi
        self.ky = np.fft.fftfreq(Ny, d=self.dy) * 2 * np.pi
        self.kx2D, self.ky2D = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.k2 = self.kx2D ** 2 + self.ky2D ** 2
        self.k2[0, 0] = 1e-14

        self.psi_hat = np.zeros((Nx, Ny), dtype=np.complex128)
        self.zeta_hat = np.zeros((Nx, Ny), dtype=np.complex128)

    def initial_conditions(self, psi_init_func) -> None:
        x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        psi0 = psi_init_func(X, Y)
        self.psi_hat = np.fft.fftn(psi0)
        self.zeta_hat = -self.k2 * self.psi_hat

    def step(self) -> None:
        psi_x_hat = 1j * self.kx2D * self.psi_hat
        self.zeta_hat = self.zeta_hat + self.dt * (-self.beta * psi_x_hat)
        self.psi_hat = -self.zeta_hat / self.k2

    def solve(self) -> np.ndarray:
        x = np.linspace(0, self.Lx, self.Nx, endpoint=False)
        y = np.linspace(0, self.Ly, self.Ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        snaps = [np.fft.ifftn(self.psi_hat).real]
        for _ in range(self.nt):
            self.step()
            snaps.append(np.fft.ifftn(self.psi_hat).real)
        return np.array(snaps)


class FlexuralBeamWave:
    """Euler–Bernoulli flexural wave in a thin beam.

    This solves ``w_tt + D * w_xxxx = 0`` with second-order differences
    in space and a leapfrog update in time.
    """

    def __init__(self, D: float = 0.01, L: float = 2.0, Nx: int = 801, dt: Optional[float] = None, T: float = 5.0) -> None:
        self.D = D
        self.L = L
        self.Nx = Nx
        self.x = np.linspace(0, L, Nx)
        self.dx = self.x[1] - self.x[0]
        if dt is None:
            dt = 0.2 * self.dx ** 2 / np.sqrt(D)
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)

        self.w_old = np.zeros(Nx)
        self.w = np.zeros(Nx)
        self.w_new = np.zeros(Nx)

    def initial_conditions(self, w_init_func) -> None:
        self.w = w_init_func(self.x)
        self.w_old[:] = self.w

    def step(self) -> None:
        for i in range(2, self.Nx - 2):
            w_xx = (self.w[i + 1] - 2 * self.w[i] + self.w[i - 1]) / self.dx ** 2
            w_xx_plus = (self.w[i + 2] - 2 * self.w[i + 1] + self.w[i]) / self.dx ** 2
            w_xx_minus = (self.w[i] - 2 * self.w[i - 1] + self.w[i - 2]) / self.dx ** 2
            w_xxxx = (w_xx_plus - 2 * w_xx + w_xx_minus) / self.dx ** 2
            self.w_new[i] = 2 * self.w[i] - self.w_old[i] - self.dt ** 2 * self.D * w_xxxx
        self.w_new[0] = 0.0
        self.w_new[1] = 0.0
        self.w_new[-1] = 0.0
        self.w_new[-2] = 0.0
        self.w_old, self.w = self.w, self.w_new

    def solve(self) -> np.ndarray:
        snaps = [self.w.copy()]
        for _ in range(self.nt):
            self.step()
            snaps.append(self.w.copy())
        return np.array(snaps)


class AlfvenWave:
    """One-dimensional Alfvén wave along a uniform magnetic field.

    The update solves ``v_tt = v_A**2 * v_xx`` with Dirichlet boundaries.
    """

    def __init__(self, B0: float = 1.0, rho: float = 1.0, mu0: float = 1.0, L: float = 2.0, Nx: int = 800, dt: Optional[float] = None, T: float = 2.0) -> None:
        self.B0 = B0
        self.rho = rho
        self.mu0 = mu0
        self.vA = B0 / np.sqrt(mu0 * rho)
        self.L = L
        self.Nx = Nx
        self.x = np.linspace(0, L, Nx)
        self.dx = self.x[1] - self.x[0]
        if dt is None:
            dt = 0.8 * self.dx / self.vA
        self.dt = dt
        self.T = T
        self.nt = int(T // dt)

        self.v_old = np.zeros(Nx)
        self.v = np.zeros(Nx)
        self.v_new = np.zeros(Nx)

    def initial_conditions(self, v_init_func) -> None:
        self.v_old = v_init_func(self.x)
        self.v = self.v_old.copy()

    def step(self) -> None:
        for i in range(1, self.Nx - 1):
            self.v_new[i] = (
                2 * self.v[i]
                - self.v_old[i]
                + (self.vA * self.dt / self.dx) ** 2 * (self.v[i + 1] - 2 * self.v[i] + self.v[i - 1])
            )
        self.v_new[0] = 0.0
        self.v_new[-1] = 0.0
        self.v_old, self.v = self.v, self.v_new

    def solve(self) -> np.ndarray:
        snaps = [self.v.copy()]
        for _ in range(self.nt):
            self.step()
            snaps.append(self.v.copy())
        return np.array(snaps)


__all__ = [
    "PWaveSimulation",
    "SWaveSimulation",
    "SHWaveSimulation",
    "SVWaveSimulation",
    "PlaneAcousticWave",
    "SphericalAcousticWave",
    "DeepWaterGravityWave",
    "ShallowWaterGravityWave",
    "CapillaryWave",
    "InternalGravityWave",
    "KelvinWave",
    "RossbyPlanetaryWave",
    "FlexuralBeamWave",
    "AlfvenWave",
]

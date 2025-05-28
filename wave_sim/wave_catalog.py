"""Wave catalog with configurations for the high quality solver."""

from __future__ import annotations

import numpy as np

from .high_quality import ConstantSpeed, PointSource, ConstantElasticSpeed
from .elastic2d import ElasticWaveSimulator2D
from .core.boundary import BoundaryCondition
from .initial_conditions import gaussian_2d

# --- NEW: analytical phase-speed helpers ------------------------------------
from .dispersion import (
    rayleigh_wave_speed,
    love_wave_dispersion,
    lamb_s0_mode,
    lamb_a0_mode,
    stoneley_wave_speed,
    scholte_wave_speed,
)

# Default elastic constants (quick-demo values)
_ALPHA = 3000.0      # m s-1  P-wave
_BETA  = 1500.0      # m s-1  S-wave
_RHO   = 2000.0      # kg m-3
_WATER_C   = 1480.0  # m s-1
_WATER_RHO = 1000.0  # kg m-3


def gaussian_initial_condition(X: np.ndarray, Y: np.ndarray, sigma: float = 5.0):
    """Return a Gaussian pulse centered in the grid."""
    return gaussian_2d(X, Y, sigma=sigma)


class Wave2DConfig:
    """Base configuration for 2-D waves using ``WaveSimulator2D``."""

    is_2d_fd = True
    default_speed = 1.0

    def __init__(self, c: float | None = None, **kwargs):
        self.c = c if c is not None else self.default_speed
        self.source_params = kwargs.get("source_params", {})
        self.initial_condition = kwargs.get("initial_condition")

    def get_scene_builder(self):
        def builder(resolution):
            w, h = resolution
            objs = [ConstantSpeed(self.c)]
            if self.initial_condition is not None:
                X, Y = np.meshgrid(np.arange(w), np.arange(h), indexing="ij")
                init = self.initial_condition(X, Y)
            else:
                init = None
            sp = self.source_params
            if sp:
                x = sp.get("x", w // 2)
                y = sp.get("y", h // 2)
                freq = sp.get("freq", 0.1)
                amp = sp.get("amplitude", 5.0)
                objs.append(PointSource(x, y, freq=freq, amplitude=amp))
            return objs, w, h, init

        return builder


class ScalarWave2D:
    """Simple finite-difference solver for the scalar 2-D wave equation."""

    def __init__(self, c: float, width: int, height: int, *, dx: float = 1.0, dt: float = 1.0,
                 boundary: BoundaryCondition = BoundaryCondition.REFLECTIVE) -> None:
        self.c = float(c)
        self.width = int(width)
        self.height = int(height)
        self.dx = float(dx)
        self.dt = float(dt)
        self.boundary = boundary

        self.u = np.zeros((self.height, self.width), dtype=float)
        self.u_prev = np.zeros_like(self.u)
        self.u_next = np.zeros_like(self.u)

    def initial_conditions(self, func) -> None:
        X, Z = np.meshgrid(np.arange(self.width) * self.dx,
                           np.arange(self.height) * self.dx, indexing="xy")
        self.u[:] = func(X, Z)
        self.u_prev[:] = self.u

    def _apply_bc(self, arr: np.ndarray) -> None:
        if self.boundary == BoundaryCondition.REFLECTIVE:
            arr[0, :] = arr[1, :]
            arr[-1, :] = arr[-2, :]
            arr[:, 0] = arr[:, 1]
            arr[:, -1] = arr[:, -2]

    def step(self) -> None:
        r = (self.c * self.dt / self.dx) ** 2
        u = self.u
        up = self.u_prev
        un = self.u_next
        un[1:-1, 1:-1] = (
            2.0 * u[1:-1, 1:-1]
            - up[1:-1, 1:-1]
            + r * (
                u[2:, 1:-1]
                + u[:-2, 1:-1]
                + u[1:-1, 2:]
                + u[1:-1, :-2]
                - 4.0 * u[1:-1, 1:-1]
            )
        )
        self._apply_bc(un)
        self.u_prev, self.u, self.u_next = self.u, un, up

    def get_field(self) -> np.ndarray:
        return self.u

    def energy(self) -> float:
        return float(np.sum(self.u ** 2))


class PrimaryWave(Wave2DConfig):
    """Compressional body wave."""

    default_speed = 3000.0


class SecondaryWave(Wave2DConfig):
    """Shear body wave."""

    default_speed = 1500.0


class SHWave(Wave2DConfig):
    """Horizontally polarised shear."""

    default_speed = 1500.0

    def get_displacement_y(self, frame: np.ndarray) -> np.ndarray:
        return frame


class SVWave(Wave2DConfig):
    """Vertically polarised shear using scalar potential."""

    default_speed = 1500.0

    def get_displacement_components(self, frame: np.ndarray, dx: float = 1.0):
        dpsi_dz, dpsi_dx = np.gradient(frame, dx, dx, edge_order=2)
        ux = dpsi_dz
        uz = -dpsi_dx
        return ux, uz


class RayleighWave:
    """Simple elastic Rayleigh surface wave solver."""

    def __init__(self, width: int = 200, height: int = 100, *, dx: float = 1.0, dt: float = 0.5) -> None:
        cp = _ALPHA
        cs = _BETA
        objects = [ConstantElasticSpeed(cp, cs)]
        self.sim = ElasticWaveSimulator2D(width, height, objects, backend="cpu", dx=dx, dt=dt)
        xp = self.sim.xp
        z = xp.arange(height) * dx
        eigen = xp.exp(-z / 20.0)
        self.sim.u[..., 1] = eigen[:, None]
        self.sim.u_prev[...] = self.sim.u

    def step(self) -> None:
        self.sim.update_field()

    def get_displacement(self) -> tuple[np.ndarray, np.ndarray]:
        ux, uz = self.sim.get_displacement()
        return np.asarray(ux), np.asarray(uz)

    def energy(self) -> float:
        ux, uz = self.get_displacement()
        return float(np.sum(ux ** 2 + uz ** 2))


class LoveWave:
    """Shear-horizontal surface wave solver using a scalar formulation."""

    def __init__(self, width: int = 200, height: int = 100, *, dx: float = 1.0, dt: float = 0.5) -> None:
        self.solver = ScalarWave2D(_BETA, width, height, dx=dx, dt=dt)
        def init_func(X, Z):
            return np.exp(-Z / 20.0)
        self.solver.initial_conditions(init_func)

    def step(self) -> None:
        self.solver.step()

    def get_displacement(self) -> np.ndarray:
        return self.solver.get_field()

    def energy(self) -> float:
        return self.solver.energy()


class LambS0Mode:
    """Symmetric Lamb mode solver using the elastic formulation."""

    def __init__(self, width: int = 200, height: int = 50, *, dx: float = 1.0, dt: float = 0.5) -> None:
        objects = [ConstantElasticSpeed(_ALPHA, _BETA)]
        self.sim = ElasticWaveSimulator2D(width, height, objects, backend="cpu", dx=dx, dt=dt)
        xp = self.sim.xp
        z = xp.arange(height) * dx
        eig = xp.cos(np.pi * (z - z.mean()) / height)
        self.sim.u[..., 1] = eig[:, None]
        self.sim.u_prev[...] = self.sim.u

    def step(self) -> None:
        self.sim.update_field()

    def get_displacement(self) -> tuple[np.ndarray, np.ndarray]:
        ux, uz = self.sim.get_displacement()
        return np.asarray(ux), np.asarray(uz)

    def energy(self) -> float:
        ux, uz = self.get_displacement()
        return float(np.sum(ux ** 2 + uz ** 2))


class LambA0Mode:
    """Antisymmetric Lamb mode solver using the elastic formulation."""

    def __init__(self, width: int = 200, height: int = 50, *, dx: float = 1.0, dt: float = 0.5) -> None:
        objects = [ConstantElasticSpeed(_ALPHA, _BETA)]
        self.sim = ElasticWaveSimulator2D(width, height, objects, backend="cpu", dx=dx, dt=dt)
        xp = self.sim.xp
        z = xp.arange(height) * dx
        eig = xp.sin(np.pi * (z - z.mean()) / height)
        self.sim.u[..., 1] = eig[:, None]
        self.sim.u_prev[...] = self.sim.u

    def step(self) -> None:
        self.sim.update_field()

    def get_displacement(self) -> tuple[np.ndarray, np.ndarray]:
        ux, uz = self.sim.get_displacement()
        return np.asarray(ux), np.asarray(uz)

    def energy(self) -> float:
        ux, uz = self.get_displacement()
        return float(np.sum(ux ** 2 + uz ** 2))


class StoneleyWave:
    """Interface wave between two elastic half spaces."""

    def __init__(self, width: int = 200, height: int = 100, *, dx: float = 1.0, dt: float = 0.5) -> None:
        self.sim = ElasticWaveSimulator2D(width, height, [], backend="cpu", dx=dx, dt=dt)
        mid = height // 2
        self.sim.c[:mid, :, 0] = _ALPHA
        self.sim.c[:mid, :, 1] = _BETA
        self.sim.c[mid:, :, 0] = _ALPHA * 0.6
        self.sim.c[mid:, :, 1] = _BETA * 0.6
        xp = self.sim.xp
        z = xp.arange(height) * dx
        eig = xp.exp(-abs(z - mid * dx) / 20.0)
        self.sim.u[..., 1] = eig[:, None]
        self.sim.u_prev[...] = self.sim.u

    def step(self) -> None:
        self.sim.update_field()

    def get_displacement(self) -> tuple[np.ndarray, np.ndarray]:
        ux, uz = self.sim.get_displacement()
        return np.asarray(ux), np.asarray(uz)

    def energy(self) -> float:
        ux, uz = self.get_displacement()
        return float(np.sum(ux ** 2 + uz ** 2))


class ScholteWave:
    """Interface wave at a fluid/solid boundary."""

    def __init__(self, width: int = 200, height: int = 100, *, dx: float = 1.0, dt: float = 0.5) -> None:
        self.sim = ElasticWaveSimulator2D(width, height, [], backend="cpu", dx=dx, dt=dt)
        mid = height // 2
        self.sim.c[:mid, :, 0] = _ALPHA
        self.sim.c[:mid, :, 1] = _BETA
        self.sim.c[mid:, :, 0] = _WATER_C
        self.sim.c[mid:, :, 1] = 1e-6
        xp = self.sim.xp
        z = xp.arange(height) * dx
        eig = xp.exp(-abs(z - mid * dx) / 20.0)
        self.sim.u[..., 1] = eig[:, None]
        self.sim.u_prev[...] = self.sim.u

    def step(self) -> None:
        self.sim.update_field()

    def get_displacement(self) -> tuple[np.ndarray, np.ndarray]:
        ux, uz = self.sim.get_displacement()
        return np.asarray(ux), np.asarray(uz)

    def energy(self) -> float:
        ux, uz = self.get_displacement()
        return float(np.sum(ux ** 2 + uz ** 2))




__all__ = [
    "PrimaryWave",
    "SecondaryWave",
    "SHWave",
    "SVWave",
    "RayleighWave",
    "LoveWave",
    "LambS0Mode",
    "LambA0Mode",
    "StoneleyWave",
    "ScholteWave",
    "gaussian_initial_condition",
]

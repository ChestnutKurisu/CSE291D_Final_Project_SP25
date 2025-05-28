"""Simplified 2-D elastic wave solver with Rayleigh surface-wave demo."""

from __future__ import annotations

import numpy as np

from .backend import get_array_module
from .core.boundary import BoundaryCondition
from .high_quality.scene_objects import ConstantElasticSpeed, SceneObject


class ElasticWaveSimulator2D:
    """Finite-difference elastic solver using vector displacement."""

    def __init__(
        self,
        width: int,
        height: int,
        scene_objects: list[SceneObject] | None = None,
        *,
        backend: str = "gpu",
        boundary: BoundaryCondition = BoundaryCondition.REFLECTIVE,
        dx: float = 1.0,
        dt: float = 1.0,
    ) -> None:
        self.xp = get_array_module(backend)
        xp = self.xp

        self.width = int(width)
        self.height = int(height)
        self.dx = float(dx)
        self.dt = float(dt)
        self.boundary = boundary
        self.scene_objects = scene_objects if scene_objects is not None else []

        self.c = xp.ones((height, width, 2), dtype=xp.float32)
        self.d = xp.ones((height, width), dtype=xp.float32)
        self.u = xp.zeros((height, width, 2), dtype=xp.float32)
        self.u_prev = xp.zeros_like(self.u)
        self.t = 0.0

        self._render_scene_properties()

    def _render_scene_properties(self) -> None:
        self.c[...] = 1.0
        self.d[...] = 1.0
        for obj in self.scene_objects:
            obj.render(self.u[..., 0], self.c, self.d)

    # ------------------------------------------------------------------
    def _laplacian(self, arr: np.ndarray) -> np.ndarray:
        xp = self.xp
        d0, d1 = xp.gradient(arr, self.dx, self.dx, edge_order=2)
        dd0 = xp.gradient(d0, self.dx, axis=0, edge_order=2)
        dd1 = xp.gradient(d1, self.dx, axis=1, edge_order=2)
        return dd0 + dd1

    def _divergence(self, ux: np.ndarray, uz: np.ndarray) -> np.ndarray:
        xp = self.xp
        dzu, dxu = xp.gradient(ux, self.dx, self.dx, edge_order=2)
        dzw, dxw = xp.gradient(uz, self.dx, self.dx, edge_order=2)
        return dxu + dzw

    def update_field(self) -> None:
        xp = self.xp
        ux = self.u[..., 0]
        uz = self.u[..., 1]
        ux_prev = self.u_prev[..., 0]
        uz_prev = self.u_prev[..., 1]
        cp = self.c[..., 0]
        cs = self.c[..., 1]

        div_u = self._divergence(ux, uz)
        grad_div_z, grad_div_x = xp.gradient(div_u, self.dx, self.dx, edge_order=2)
        lap_ux = self._laplacian(ux)
        lap_uz = self._laplacian(uz)

        accel_x = (cp ** 2 - cs ** 2) * grad_div_x + cs ** 2 * lap_ux
        accel_z = (cp ** 2 - cs ** 2) * grad_div_z + cs ** 2 * lap_uz

        vx = (ux - ux_prev) * self.d
        vz = (uz - uz_prev) * self.d

        ux_next = ux + vx + accel_x * (self.dt ** 2)
        uz_next = uz + vz + accel_z * (self.dt ** 2)

        self.u_prev[..., 0] = ux
        self.u_prev[..., 1] = uz
        self.u[..., 0] = ux_next
        self.u[..., 1] = uz_next
        self.t += self.dt

    # ------------------------------------------------------------------
    def get_displacement(self) -> tuple[np.ndarray, np.ndarray]:
        return self.u[..., 0], self.u[..., 1]


# ----------------------------------------------------------------------
# Example driver -------------------------------------------------------

def rayleigh_surface_demo(width: int = 200, height: int = 100) -> ElasticWaveSimulator2D:
    """Return a simulator instance configured for a simple Rayleigh-wave demo."""

    objects = [ConstantElasticSpeed(3000.0, 1500.0)]
    sim = ElasticWaveSimulator2D(width, height, objects, dx=1.0, dt=0.5, backend="cpu")

    xp = sim.xp
    z = xp.arange(height) * sim.dx
    eigen = xp.exp(-z / 20.0)
    sim.u[..., 1] = eigen[:, None]
    sim.u_prev[...] = sim.u
    return sim

__all__ = ["ElasticWaveSimulator2D", "rayleigh_surface_demo", "ConstantElasticSpeed"]

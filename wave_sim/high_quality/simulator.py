import numpy as np
import warnings
import scipy.signal

from ..backend import get_array_module
from ..core.boundary import BoundaryCondition
from ..core.kernels import get_laplacian_kernel


class SceneObject:
    """Interface for simulation scene objects.

    Attributes
    ----------
    is_static : bool
        If ``True`` the object's contribution to medium properties does not
        change over time.  Static objects only need to be rendered once when
        the scene is (re)initialized.
    """

    #: Flag indicating whether the object's render output is time invariant.
    is_static: bool = True

    def render(self, field, wave_speed_field, dampening_field):
        """Render the object's effect on wave speed and dampening fields."""
        pass

    def update_field(self, field, t):
        """Modify the wave field at time ``t``."""
        pass

    def render_visualization(self, image):
        """Draw a visualization representation of the object."""
        pass


class WaveSimulator2D:
    """GPU accelerated 2-D wave equation solver.

    Parameters
    ----------
    width, height : int
        Domain size in pixels.
    scene_objects : list, optional
        Objects placed in the scene.
    initial_field : array-like, optional
        Initial displacement field.
    backend : {"gpu", "cpu"}, optional
        Array backend. ``gpu`` uses :mod:`cupy` when available.
    boundary : :class:`~wave_sim.core.boundary.BoundaryCondition` or str, optional
        Boundary condition.
    dx : float, optional
        Spatial resolution of the grid.  ``laplacian`` is scaled by ``1/dx^2``.
    sponge_thickness : int, optional
        Thickness of the absorbing sponge layer when ``boundary`` is
        ``ABSORBING``.  A value of 0 disables the sponge.
    """

    def __init__(
        self,
        width,
        height,
        scene_objects=None,
        initial_field=None,
        backend="gpu",
        boundary=BoundaryCondition.REFLECTIVE,
        dx=1.0,
        dt=1.0,
        sponge_thickness: int = 8,
        elastic: bool = False,
    ):
        self.xp = get_array_module(backend)
        xp = self.xp

        self.global_dampening = 1.0
        if isinstance(boundary, BoundaryCondition):
            self.boundary = boundary
        else:
            self.boundary = BoundaryCondition(boundary)
        self.elastic = bool(elastic)
        if self.elastic:
            self.c = xp.ones((height, width, 2), dtype=xp.float32)
            self.u = xp.zeros((height, width, 2), dtype=xp.float32)
            self.u_prev = xp.zeros_like(self.u)
        else:
            self.c = xp.ones((height, width), dtype=xp.float32)
            self.u = xp.zeros((height, width), dtype=xp.float32)
            self.u_prev = xp.zeros_like(self.u)
        self.d = xp.ones((height, width), dtype=xp.float32)

        if initial_field is not None:
            self.u[:] = initial_field
            self.u_prev[:] = initial_field

        self.laplacian_kernel = get_laplacian_kernel(xp)

        self.t = 0.0
        self.dt = dt
        self.dx = dx
        self.sponge_thickness = int(sponge_thickness)
        self.scene_objects = scene_objects if scene_objects is not None else []

        # determine whether scene needs to be rendered every frame
        self.scene_static = all(getattr(o, "is_static", True) for o in self.scene_objects)
        self._scene_dirty = True
        self._render_scene_properties()
        if xp.max(self.c) * self.dt / self.dx > 0.7:
            warnings.warn(
                f"Potential CFL violation: max(c) * dt / dx = {float(xp.max(self.c) * self.dt / self.dx):.2f} > 0.7."
            )

    def reset_time(self):
        self.t = 0.0

    def _render_scene_properties(self):
        """Render wave speed and dampening fields from scene objects."""
        self.c.fill(1.0)
        self.d.fill(1.0)
        for obj in self.scene_objects:
            if self.elastic:
                obj.render(self.u[..., 0], self.c, self.d)
            else:
                obj.render(self.u, self.c, self.d)
        self._scene_dirty = False

    def _update_field_scalar(self):
        xp = self.xp
        if self.boundary == BoundaryCondition.PERIODIC:
            bmode = "wrap"
        elif self.boundary == BoundaryCondition.REFLECTIVE:
            bmode = "symm"
        else:
            bmode = "fill"
        if xp.__name__ == "cupy":
            import cupyx.scipy.signal  # type: ignore
            laplacian = xp.asarray(
                cupyx.scipy.signal.convolve2d(
                    self.u, self.laplacian_kernel, mode="same", boundary=bmode
                )
            )
        else:
            laplacian = scipy.signal.convolve2d(
                self.u, self.laplacian_kernel, mode="same", boundary=bmode
            )
        v = (self.u - self.u_prev) * self.d * self.global_dampening
        r = self.u + v + laplacian * (self.c * self.dt / self.dx) ** 2
        if self.boundary == BoundaryCondition.ABSORBING and self.sponge_thickness > 0:
            n = self.sponge_thickness
            if n > 0 and min(self.u.shape) > 2 * n:
                taper = xp.sin(0.5 * xp.pi * xp.linspace(0, 1, n)) ** 2

                self.u_prev[:n, :] *= taper[::-1, None]
                self.u[:n, :] *= taper[::-1, None]

                self.u_prev[-n:, :] *= taper[:, None]
                self.u[-n:, :] *= taper[:, None]

                self.u_prev[:, :n] *= taper[None, ::-1]
                self.u[:, :n] *= taper[None, ::-1]

                self.u_prev[:, -n:] *= taper[None, :]
                self.u[:, -n:] *= taper[None, :]
            elif n > 0:
                warnings.warn(
                    f"Sponge thickness {n} is too large for domain size {self.u.shape}. Disabling sponge."
                )
        self.u_prev[:] = self.u
        self.u[:] = r
        self.t += self.dt

    def _update_field_elastic(self):
        xp = self.xp
        ux = self.u[..., 0]
        uz = self.u[..., 1]
        ux_prev = self.u_prev[..., 0]
        uz_prev = self.u_prev[..., 1]
        cp = self.c[..., 0]
        cs = self.c[..., 1]

        def lap(arr):
            d0, d1 = xp.gradient(arr, self.dx, self.dx, edge_order=2)
            dd0 = xp.gradient(d0, self.dx, axis=0, edge_order=2)
            dd1 = xp.gradient(d1, self.dx, axis=1, edge_order=2)
            return dd0 + dd1

        duz_dz, dux_dx = xp.gradient(ux, self.dx, self.dx, edge_order=2)
        dvz_dz, dvx_dx = xp.gradient(uz, self.dx, self.dx, edge_order=2)
        div_u = dvx_dx + duz_dz
        ddiv_dz, ddiv_dx = xp.gradient(div_u, self.dx, self.dx, edge_order=2)

        accel_x = (cp ** 2 - cs ** 2) * ddiv_dx + cs ** 2 * lap(ux)
        accel_z = (cp ** 2 - cs ** 2) * ddiv_dz + cs ** 2 * lap(uz)

        vx = (ux - ux_prev) * self.d * self.global_dampening
        vz = (uz - uz_prev) * self.d * self.global_dampening

        ux_next = ux + vx + accel_x * (self.dt ** 2)
        uz_next = uz + vz + accel_z * (self.dt ** 2)

        if self.boundary == BoundaryCondition.ABSORBING and self.sponge_thickness > 0:
            n = self.sponge_thickness
            if n > 0 and min(self.u.shape[:2]) > 2 * n:
                taper = xp.sin(0.5 * xp.pi * xp.linspace(0, 1, n)) ** 2
                for arr in (ux_prev, uz_prev, ux, uz):
                    arr[:n, :] *= taper[::-1, None]
                    arr[-n:, :] *= taper[:, None]
                    arr[:, :n] *= taper[None, ::-1]
                    arr[:, -n:] *= taper[None, :]
            elif n > 0:
                warnings.warn(
                    f"Sponge thickness {n} is too large for domain size {self.u.shape}. Disabling sponge."
                )

        self.u_prev[..., 0] = ux
        self.u_prev[..., 1] = uz
        self.u[..., 0] = ux_next
        self.u[..., 1] = uz_next
        self.t += self.dt

    def update_field(self):
        if self.elastic:
            self._update_field_elastic()
        else:
            self._update_field_scalar()

    def update_scene(self):
        if not self.scene_static or self._scene_dirty:
            self._render_scene_properties()
        for obj in self.scene_objects:
            if self.elastic:
                obj.update_field(self.u[..., 0], self.t)
            else:
                obj.update_field(self.u, self.t)

    def mark_scene_dirty(self):
        """Flag the scene so medium properties are re-rendered on next update."""
        self.scene_static = all(getattr(o, "is_static", True) for o in self.scene_objects)
        self._scene_dirty = True

    def get_field(self):
        return self.u

    def render_visualization(self, image=None):
        xp = self.xp
        if image is None:
            image = np.zeros((self.c.shape[0], self.c.shape[1], 3), dtype=np.uint8)
        for obj in self.scene_objects:
            obj.render_visualization(image)
        return image

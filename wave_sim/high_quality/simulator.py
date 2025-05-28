import numpy as np
import warnings
import scipy.signal

from ..backend import get_array_module
from ..core.boundary import BoundaryCondition
from ..core.kernels import get_laplacian_kernel


class SceneObject:
    """Interface for simulation scene objects."""

    def render(self, field, wave_speed_field, dampening_field):
        pass

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
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
    ):
        self.xp = get_array_module(backend)
        xp = self.xp

        self.global_dampening = 1.0
        if isinstance(boundary, BoundaryCondition):
            self.boundary = boundary
        else:
            self.boundary = BoundaryCondition(boundary)
        self.c = xp.ones((height, width), dtype=xp.float32)
        self.d = xp.ones((height, width), dtype=xp.float32)
        self.u = xp.zeros((height, width), dtype=xp.float32)
        self.u_prev = xp.zeros((height, width), dtype=xp.float32)

        if initial_field is not None:
            self.u[:] = initial_field
            self.u_prev[:] = initial_field

        self.laplacian_kernel = get_laplacian_kernel(xp)

        self.t = 0.0
        self.dt = dt
        self.dx = dx
        self.sponge_thickness = int(sponge_thickness)
        self.scene_objects = scene_objects if scene_objects is not None else []

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
            obj.render(self.u, self.c, self.d)

    def update_field(self):
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
        if self.boundary == BoundaryCondition.ABSORBING:
            n = 32  # sponge width
            taper = xp.sin(0.5 * xp.pi * xp.linspace(0, 1, n)) ** 2
            self.u_prev[:n] *= taper[::-1, None]
            self.u[:n] *= taper[::-1, None]
            self.u_prev[-n:] *= taper[:, None]
            self.u[-n:] *= taper[:, None]
            self.u_prev[:, :n] *= taper[None, ::-1]
            self.u[:, :n] *= taper[None, ::-1]
            self.u_prev[:, -n:] *= taper[None, :]
            self.u[:, -n:] *= taper[None, :]
        self.u_prev[:] = self.u
        self.u[:] = r
        self.t += self.dt

    def update_scene(self):
        self._render_scene_properties()
        for obj in self.scene_objects:
            obj.update_field(self.u, self.t)

    def get_field(self):
        return self.u

    def render_visualization(self, image=None):
        xp = self.xp
        if image is None:
            image = np.zeros((self.c.shape[0], self.c.shape[1], 3), dtype=np.uint8)
        for obj in self.scene_objects:
            obj.render_visualization(image)
        return image

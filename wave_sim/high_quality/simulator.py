import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.signal
except Exception:  # pragma: no cover - optional dependency
    cp = None

import scipy.signal


class SceneObject:
    """Interface for simulation scene objects."""

    def render(self, field, wave_speed_field, dampening_field):
        pass

    def update_field(self, field, t):
        pass

    def render_visualization(self, image):
        pass


class WaveSimulator2D:
    """GPU accelerated 2-D wave equation solver."""

    def __init__(self, width, height, scene_objects=None, initial_field=None,
                 backend="gpu", boundary="reflective"):
        if backend == "gpu" and cp is not None:
            self.xp = cp
        else:
            self.xp = np
        xp = self.xp

        self.global_dampening = 1.0
        self.boundary = boundary
        self.c = xp.ones((height, width), dtype=xp.float32)
        self.d = xp.ones((height, width), dtype=xp.float32)
        self.u = xp.zeros((height, width), dtype=xp.float32)
        self.u_prev = xp.zeros((height, width), dtype=xp.float32)

        if initial_field is not None:
            self.u[:] = initial_field
            self.u_prev[:] = initial_field

        self.laplacian_kernel = xp.array([[0.066, 0.184, 0.066],
                                          [0.184, -1.0, 0.184],
                                          [0.066, 0.184, 0.066]], dtype=xp.float32)

        self.t = 0.0
        self.dt = 1.0
        self.scene_objects = scene_objects if scene_objects is not None else []

    def reset_time(self):
        self.t = 0.0

    def update_field(self):
        xp = self.xp
        if self.boundary == "periodic":
            bmode = "wrap"
        elif self.boundary == "reflective":
            bmode = "symm"
        else:
            bmode = "fill"
        if xp is cp:
            laplacian = cp.asarray(
                cupyx.scipy.signal.convolve2d(
                    self.u, self.laplacian_kernel, mode="same", boundary=bmode
                )
            )
        else:
            laplacian = scipy.signal.convolve2d(
                self.u, self.laplacian_kernel, mode="same", boundary=bmode
            )
        v = (self.u - self.u_prev) * self.d * self.global_dampening
        r = self.u + v + laplacian * (self.c * self.dt) ** 2
        if self.boundary == "absorbing":
            damp = 8
            for i in range(damp):
                factor = (damp - i) / damp
                r[i, :] *= factor
                r[-1 - i, :] *= factor
                r[:, i] *= factor
                r[:, -1 - i] *= factor
        self.u_prev[:] = self.u
        self.u[:] = r
        self.t += self.dt

    def update_scene(self):
        self.c.fill(1.0)
        self.d.fill(1.0)
        for obj in self.scene_objects:
            obj.render(self.u, self.c, self.d)
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

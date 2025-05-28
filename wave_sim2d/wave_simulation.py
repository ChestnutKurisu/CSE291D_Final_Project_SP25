import cupy as cp
import cupyx.scipy.signal
import numpy as np
from typing import Optional

from .scene_objects.base import SceneObject


class WaveSimulator2D:
    """2D wave equation solver using GPU arrays via CuPy."""

    def __init__(
        self,
        width: int,
        height: int,
        scene_objects: Optional[list[SceneObject]] = None,
        dt: float = 1.0,
        global_dampening: float = 1.0,
        laplacian_kernel: Optional[cp.ndarray] = None,
    ):
        self.width = width
        self.height = height
        self.dt = dt
        self.global_dampening = global_dampening

        self.c = cp.ones((height, width), dtype=cp.float32)
        self.d = cp.ones((height, width), dtype=cp.float32)
        self.u = cp.zeros((height, width), dtype=cp.float32)
        self.u_prev = cp.zeros((height, width), dtype=cp.float32)

        if laplacian_kernel is None:
            laplacian_kernel = cp.array(
                [
                    [0.066, 0.184, 0.066],
                    [0.184, -1.0, 0.184],
                    [0.066, 0.184, 0.066],
                ],
                dtype=cp.float32,
            )
        self.laplacian_kernel = laplacian_kernel

        self.t = 0.0
        self.scene_objects = scene_objects or []

    def reset_time(self):
        self.t = 0.0

    def update_scene(self):
        self.c.fill(1.0)
        self.d.fill(1.0)
        for obj in self.scene_objects:
            obj.render(self.u, self.c, self.d)
        for obj in self.scene_objects:
            obj.update_field(self.u, self.t)

    def update_field(self):
        laplacian = cupyx.scipy.signal.convolve2d(
            self.u, self.laplacian_kernel, mode="same", boundary="fill"
        )
        v = (self.u - self.u_prev) * self.d * self.global_dampening
        r = self.u + v + laplacian * (self.c * self.dt) ** 2

        self.u_prev[:] = self.u
        self.u[:] = r
        self.t += self.dt

    def get_field(self) -> cp.ndarray:
        return self.u

    def run_simulation(self, total_steps: int):
        for _ in range(total_steps):
            self.update_scene()
            self.update_field()

import cupy as cp
import numpy as np

from .base import SceneObject


class StaticRefractiveIndex(SceneObject):
    """Static refractive index field. c = 1/n."""

    def __init__(self, refractive_index_field: np.ndarray):
        clipped = np.clip(refractive_index_field, 0.9, 10.0)
        self.c_field = cp.array(1.0 / clipped, dtype=cp.float32)

    def render(self, field: cp.ndarray, wave_speed_field: cp.ndarray, dampening_field: cp.ndarray):
        wave_speed_field[:] = self.c_field

    def update_field(self, field: cp.ndarray, t: float):
        pass
